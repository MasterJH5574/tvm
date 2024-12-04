from pathlib import Path
from typing import Optional

# isort: off
import torch
from torch.utils import dlpack
from frontend.compile import compile
from frontend.utils import SetConfig

from mlc_llm.compiler_pass.blas_dispatch import BLASDispatch
from mlc_llm.compiler_pass.clean_up_tir_attrs import CleanUpTIRAttrs
from mlc_llm.compiler_pass.fuse_transpose_matmul import FuseTransposeMatmul
from mlc_llm.compiler_pass.lift_global_buffer_alloc import LiftTIRGlobalBufferAlloc

# isort: on
import tvm
from tvm import IRModule
from tvm import dlight as dl

from fx_translator import from_fx


@tvm.transform.module_pass(opt_level=0, name="DebugDump")
class _DebugDump:  # pylint: disable=too-few-public-methods
    """A dummy compiler pass that does nothing but logging.
    Only enabled when debug_dump is not None"""

    def __init__(self, file_name: str, show_meta: bool = False):
        self.file_name = file_name
        self.file_path = Path("debug")
        self.show_meta = show_meta

    def transform_module(
        self, mod: IRModule, _ctx: tvm.transform.PassContext
    ) -> IRModule:
        """A dummy transformation that dumps the module to file"""
        if self.file_path is not None:
            # NOTE: We use debug level here to avoid spamming the console
            print(f"Dumping IR to {self.file_path / self.file_name}")
            with open(self.file_path / self.file_name, "w", encoding="utf-8") as f:
                f.write(mod.script(show_meta=self.show_meta))
        return mod


def magpy_pipeline(target: tvm.target.Target):
    # variable_bounds = {}

    print(f"target = {target}")

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(
        mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext
    ) -> tvm.ir.IRModule:
        seq = tvm.transform.Sequential(
            [
                # Phase 0. Add additional information for compilation and remove unused Relax func
                # AttachMemoryPlanAttr(),
                # tvm.tir.transform.BindTarget(
                #     tvm.target.Target.current(allow_none=False)
                # ),
                _DebugDump("debug-phase0.py", show_meta=False),
                # Phase 1. Passes on high-level operator graph
                BLASDispatch(target),
                FuseTransposeMatmul(),
                _DebugDump("debug-phase1.py", show_meta=False),
                # Phase 2. Lowering to TIR, inherited TVM Relax's official "zero" pipeline
                tvm.relax.transform.LegalizeOps(),
                tvm.relax.transform.AnnotateTIROpPattern(),
                tvm.relax.transform.FoldConstant(),
                tvm.relax.transform.FuseOps(),
                tvm.relax.transform.FuseTIR(),
                _DebugDump("debug-phase2.py", show_meta=False),
                # Phase 3. Passes on TIR
                tvm.relax.transform.DeadCodeElimination(),
                CleanUpTIRAttrs(["op_pattern"]),
                _DebugDump("debug-phase3.py", show_meta=False),
                # Phase 4. Low-level Optimizations
                dl.ApplyDefaultSchedule(
                    dl.gpu.Matmul(),
                    dl.gpu.GEMV(),
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback(),
                ),
                LiftTIRGlobalBufferAlloc(),
                _DebugDump("debug-phase4.py", show_meta=False),
            ]
        )
        mod = seq(mod)
        return mod

    return _pipeline


def relax_dynamo(pipeline: Optional[tvm.transform.Pass] = None):

    def _relax_backend(graph_module, example_inputs):
        print("start relax backend")
        target = tvm.target.Target.current()
        dev = tvm.device(str(target.kind))
        # device = device_from_inputs(example_inputs)

        import torch  # type: ignore[import]

        assert isinstance(graph_module, torch.fx.GraphModule)

        def to_torch_tensor(nd_tensor):
            """A helper function to transfer a NDArray to torch.tensor."""
            if isinstance(nd_tensor, tvm.nd.NDArray):
                return torch.from_numpy(nd_tensor.numpy())
            elif isinstance(nd_tensor, tvm.ir.Array):
                return tuple(to_torch_tensor(x) for x in nd_tensor)
            else:
                raise ValueError(f"Unsupported type {type(nd_tensor)}")

        def to_tvm_tensor(torch_tensor: torch.Tensor):
            """A helper function to transfer a torch.tensor to NDArray."""
            if not isinstance(torch_tensor, torch._subclasses.fake_tensor.FakeTensor):
                return (
                    tvm.nd.from_dlpack(dlpack.to_dlpack(torch_tensor))
                    if torch_tensor.device.type == "cuda"
                    else tvm.nd.array(torch_tensor.numpy(), device=dev)
                )
            # Fake Tensor
            real_tensor = torch.randn(torch_tensor.shape, dtype=torch_tensor.dtype)
            return tvm.nd.array(real_tensor.numpy())

        graph_module.graph.eliminate_dead_code()

        assert len(example_inputs)

        fake_inputs = []
        if isinstance(example_inputs[0], torch._subclasses.fake_tensor.FakeTensor):
            # Fake tensors
            fake_inputs = example_inputs
        else:
            # Real tensors
            for node in graph_module.graph.nodes:
                if node.op != "placeholder":
                    continue
                if "grapharg" not in node.meta:
                    continue
                fake_tensor = node.meta["grapharg"].fake_tensor
                if fake_tensor is None:
                    continue
                fake_inputs.append(fake_tensor)

        input_info = []
        shape_vars = {}
        for tensor in fake_inputs:
            shape = []
            for s in tensor.shape:
                if isinstance(s, torch.SymInt):
                    if str(s) not in shape_vars:
                        shape_vars[str(s)] = tvm.tir.Var(str(s), "int64")
                    shape.append(shape_vars[str(s)])
                else:
                    shape.append(s)
            input_info.append((shape, tensor.dtype))

        mod = from_fx(graph_module, input_info)
        print(f"init mod = {mod}")

        # invoke optimization pipeline.
        if pipeline is None:
            # get default pipeline
            seq = tvm.relax.get_pipeline()
        elif isinstance(pipeline, str):
            # lookup by name
            seq = tvm.relax.get_pipeline(pipeline)
        else:
            seq = pipeline

        mod = mod.with_attr("target", target)
        mod = seq(mod)

        ex = tvm.relax.build(mod, target=target)

        vm = tvm.relax.VirtualMachine(ex.mod, device=dev)

        def exec_tvm(*i_args):
            print(f"start relax run")
            args = [a.contiguous() for a in i_args if isinstance(a, torch.Tensor)]
            vm_args = list()
            for arg in args:
                if arg.dim() != 0:
                    if arg.requires_grad:
                        arg = arg.detach()
                    vm_args.append(to_tvm_tensor(arg))
            outputs = vm["main"](*vm_args)
            print(f"finish relax run")
            return to_torch_tensor(outputs)

        print("finish relax backend")
        return exec_tvm

    return _relax_backend


class Example(torch.nn.Module):

    def __init__(self):
        super(Example, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


with torch.no_grad():
    model = Example().eval()
    x = torch.randn(1, 3, 4, 4)
    expect_output = model(x)
    print("expect:", expect_output)

    # set the graph compiler to inductor
    # backend = "inductor"
    # backend = relax_dynamo()
    device = tvm.cuda()
    target = tvm.target.Target.from_device(device)
    print(target)
    backend = relax_dynamo(magpy_pipeline(target=target))
    with target:
        with SetConfig({"backend": backend}):
            compiled = compile(model)
            # run the python code to compile the model. The fx graph and the guards will be printed out
            output1 = compiled(x)
            print("output1:", output1)

            # run the compiled model. "guard cache hit" means we find the compiled record and use it directly
            output2 = compiled(x)
            print("output2", output2)
            assert torch.allclose(expect_output, output1)
            assert torch.allclose(expect_output, output2)
