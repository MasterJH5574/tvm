# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm
import tvm.testing
import tvm.relay.testing
import tvm.meta_schedule as ms

from tvm import relax
from tvm import transform
from tvm.meta_schedule import tune_relax, EvolutionarySearchConfig
from tvm.meta_schedule.builder import BuilderInput, LocalBuilder
from tvm.meta_schedule.runner import RunnerInput, EvaluatorConfig, RPCRunner

from tvm.script import relax as R
from tvm.relax.testing import relay_translator

import os
import itertools
import logging
import argparse
import numpy as np
from pathlib import Path


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model",
        type=str,
        required=True,
    )
    args.add_argument(
        "--target",
        type=str,
        required=True,
    )
    args.add_argument(
        "--device",
        type=str,
        required=True,
    )
    args.add_argument(
        "--num-trials",
        type=int,
        required=True,
    )
    args.add_argument(
        "--work-dir",
        type=str,
        required=True,
    )
    args.add_argument(
        "--rpc-host",
        type=str,
        required=True,
    )
    args.add_argument(
        "--rpc-port",
        type=int,
        required=True,
    )
    args.add_argument(
        "--rpc-key",
        type=str,
        required=True,
    )
    parsed = args.parse_args()
    parsed.target = tvm.target.Target(parsed.target)
    if parsed.target.attrs.get("mtriple", None) == "aarch64-linux-gnu":
        parsed.alloc_repeat = 3
    else:
        parsed.alloc_repeat = 1
    parsed.rpc_config = ms.runner.RPCConfig(
        tracker_host=parsed.rpc_host,
        tracker_port=parsed.rpc_port,
        tracker_key=parsed.rpc_key,
        session_timeout_sec=30,
    )
    parsed.rpc_workers = parsed.rpc_config.count_num_servers(allow_missing=False)
    parsed.device = tvm.cpu() if parsed.device == "cpu" else tvm.cuda()
    return parsed


logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)
ARGS = _parse_args()


def f_build(mod, target, params):
    with transform.PassContext(opt_level=3):
        executable, mod = relax.vm.build(mod=mod, target=target)
    mod.relax_executable = executable
    return mod


def f_upload_module(session, local_path, remote_path):
    session.upload(local_path, remote_path)
    rt_mod = session.load_module(remote_path)
    exec_path = os.path.join(str(Path(local_path).parent.absolute()), "exec.tmp")
    rt_mod.relax_executable = relax.vm.load_exec_from_file(exec_path)
    return rt_mod


def f_run_evaluator(session, rt_mod, device, evaluator_config, repeated_args):
    executable = rt_mod.relax_executable
    mod = rt_mod
    vm = relax.vm.VirtualMachine(exec=executable, device=device, mod=mod)
    evaluator = vm.module.time_evaluator(
        func_name="main",
        dev=device,
        number=evaluator_config.number,
        repeat=evaluator_config.repeat,
        min_repeat_ms=evaluator_config.min_repeat_ms,
        f_preproc="cache_flush_cpu_non_first_arg"
        if evaluator_config.enable_cpu_cache_flush
        else "",
    )
    repeated_costs = []
    for args in repeated_args:
        profile_result = evaluator(*args)
        repeated_costs.append(profile_result.results)
    costs = [float(cost) for cost in itertools.chain.from_iterable(repeated_costs)]
    return costs


def main():
    task_name = ARGS.model
    work_dir = ARGS.work_dir

    path_workload = os.path.join(work_dir, f"{task_name}_database_workload.json")
    path_tuning_record = os.path.join(work_dir, f"{task_name}_database_tuning_record.json")
    database = ms.database.JSONDatabase(
        path_workload=path_workload,
        path_tuning_record=path_tuning_record,
    )

    num_layers = 18
    batch_size = 1
    image_shape = (3, 224, 224)
    input_shape = (batch_size,) + image_shape

    relay_mod, params = tvm.relay.testing.resnet.get_workload(
        num_layers=num_layers, image_shape=image_shape, batch_size=batch_size, dtype="float32"
    )

    # translate the ResNet model from Relay to Relax
    relax_mod = relay_translator.from_relay(relay_mod["main"])
    assert isinstance(relax_mod, tvm.IRModule)

    # print(R.parser.astext(relax_mod))

    # tune_relax(
    #     mod=relax_mod,
    #     target=ARGS.target,
    #     config=EvolutionarySearchConfig(
    #         num_trials_per_iter=64,
    #         num_trials_total=ARGS.num_trials,
    #     ),
    #     runner=ms.runner.RPCRunner(
    #         rpc_config=ARGS.rpc_config,
    #         alloc_repeat=3,
    #         max_workers=ARGS.rpc_workers,
    #     ),
    #     database=database,
    #     task_name=task_name,
    #     work_dir=work_dir,
    #     num_threads=os.cpu_count(),
    # )

    builder = LocalBuilder(f_build=f_build)
    builder_input = BuilderInput(mod=relax_mod, target=ARGS.target, params=params)
    builder_result = builder.build([builder_input])[0]
    assert builder_result.error_msg is None, builder_result.error_msg
    assert builder_result.artifact_path is not None

    exec_path = os.path.join(str(Path(builder_result.artifact_path).parent.absolute()), "exec.tmp")
    with transform.PassContext(opt_level=0):
        executable, _ = relax.vm.build(relax_mod, ARGS.target)
    executable.save_to_file(exec_path)

    args_info = [ms.arg_info.TensorInfo("float32", input_shape)]
    for param in params.values():
        args_info.append(ms.arg_info.TensorInfo(dtype=param.dtype, shape=param.shape))
    runner_input = RunnerInput(
        artifact_path=builder_result.artifact_path,
        device_type=ARGS.target.kind.name,
        args_info=args_info,
    )

    evaluator_config = EvaluatorConfig(
        number=1,
        repeat=10,
        min_repeat_ms=100,
        enable_cpu_cache_flush=False,
    )
    runner = RPCRunner(
        rpc_config=ARGS.rpc_config,
        evaluator_config=evaluator_config,
        alloc_repeat=3,
        max_workers=ARGS.rpc_workers,
        f_upload_module=f_upload_module,
        f_run_evaluator=f_run_evaluator,
    )

    runner_future = runner.run([runner_input])[0]
    runner_result = runner_future.result()
    assert runner_result is not None
    assert runner_result.error_msg is None, runner_result.error_msg
    assert runner_result.run_secs is not None

    for result in runner_result.run_secs:
        if isinstance(result, tvm.tir.FloatImm):
            result = result.value
        print(result)

    # with transform.PassContext(opt_level=0):
    #     ex_untuned, lib_untuned = relax.vm.build(relax_mod, ARGS.target)

    # with transform.PassContext(opt_level=3):
    #     relax_mod_best = relax.transform.MetaScheduleApplyHistoryBest(database, ARGS.target)(
    #         relax_mod
    #     )
    #     # print(R.parser.astext(relax_mod_best))
    #     ex_tuned, lib_tuned = relax.vm.build(relax_mod_best, ARGS.target)

    # vm_untuned = relax.vm.VirtualMachine(ex_untuned, ARGS.device, mod=lib_untuned)
    # vm_tuned = relax.vm.VirtualMachine(ex_tuned, ARGS.device, mod=lib_tuned)

    # data = tvm.nd.array(np.random.randn(*input_shape).astype("float32"), ARGS.device)

    # def run_and_measure(vm: relax.vm.VirtualMachine, data, params):
    #     res = vm["main"](data, *list(params.values()))
    #     evaluator = vm.module.time_evaluator("main", ARGS.device, number=50)
    #     duration = evaluator(data, *list(params.values()))
    #     return res, duration

    # res_untuned, time_untuned = run_and_measure(vm_untuned, data, params)
    # res_tuned, time_tuned = run_and_measure(vm_tuned, data, params)

    # tvm.testing.assert_allclose(res_tuned.numpy(), res_untuned.numpy(), rtol=1e-4, atol=1e-4)

    # print(f"untuned resnet:\n{time_untuned}")
    # print(f"  tuned resnet:\n{time_tuned}")


if __name__ == "__main__":
    main()
