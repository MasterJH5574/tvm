from dgl.heterograph import DGLHeteroGraph
import tvm
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import numpy as np
import dgl
import dgl.function as fn
import torch as th
from tvm.script import tir as T
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset


def get_dataset_by_name(name: str):
    if name == 'aifb':
        return AIFBDataset()
    elif name == 'mutag':
        return MUTAGDataset()
    elif name == 'bgs':
        return BGSDataset()
    elif name == 'am':
        return AMDataset()
    else:
        raise KeyError("Unknown dataset {}.".format(name))


class TorchOpTimer(object):
    def __enter__(self):
        self.start_event = th.cuda.Event(enable_timing=True)
        self.end_event = th.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, type, value, traceback):
        self.end_event.record()
        th.cuda.synchronize()  # Wait for the events to be recorded!
        self.time = self.start_event.elapsed_time(self.end_event)


def prepare_hetero_graph_simplified(g: dgl.DGLHeteroGraph):
    ntype_pointer = np.cumsum(
        [0] + [g.number_of_nodes(ntype) for ntype in g.ntypes])

    etype_pointer = [0]
    for etype in g.canonical_etypes:
        g_sub = g[etype]
        etype_pointer.append(etype_pointer[-1] + g_sub.num_edges())

    return{"ntype_node_pointer": th.IntTensor(ntype_pointer), "etype_edge_pointer": th.IntTensor(etype_pointer)}


@T.prim_func
def rgcn_hetero_forward(
    w: T.handle,
    x: T.handle,
    y: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    indptr_j: T.handle,
    indices_j: T.handle,
    n: T.int32,
    r: T.int32,
    feat_size: T.int32,
    nnz_i: T.int32,
    nnz_j: T.int32
):
    N = T.dense_fixed(n)
    R = T.dense_fixed(r)
    I = T.sparse_variable(R, (n, nnz_i), (indptr_i, indices_i), "int32")
    J = T.sparse_variable(I, (n, nnz_j), (indptr_j, indices_j), "int32")
    F_in = T.dense_fixed(feat_size)
    F_out = T.dense_fixed(feat_size)
    W = T.match_sparse_buffer(w, (R, F_out, F_in), "float32")
    X = T.match_sparse_buffer(x, (N, F_in), "float32")
    Y = T.match_sparse_buffer(y, (N, R, F_out), "float32")
    with T.iter([R, I, F_out, J, F_in], "SSSRR", "rgcn-hetero-forward") as [
        vr, vi, vout, vj, vin
    ]:
        with T.init():
            Y[vi, vr, vout] = 0.
        Y[vi, vr, vout] = Y[vi, vr, vout] + W[vr, vout, vin] * X[vj, vin]


@T.prim_func
def func(w: T.handle, x: T.handle, y: T.handle, indptr_i: T.handle, indices_i: T.handle, indptr_j: T.handle, indices_j: T.handle, n: T.int32, r: T.int32, feat_size: T.int32, nnz_i: T.int32, nnz_j: T.int32) -> None:
    W_data = T.match_buffer(w, [r * feat_size * feat_size], dtype="float32")
    X_data = T.match_buffer(x, [n * feat_size], dtype="float32")
    Y_data = T.match_buffer(y, [n * r * feat_size], dtype="float32")
    I_indptr = T.match_buffer(indptr_i, [r + 1], dtype="int32")
    I_indices = T.match_buffer(indices_i, [nnz_i], dtype="int32")
    J_indptr = T.match_buffer(indptr_j, [nnz_i + 1], dtype="int32")
    J_indices = T.match_buffer(indices_j, [nnz_j], dtype="int32")
    # body
    # with T.block("root")
    for v_vr in T.serial(r):
        with T.block("rgcn-hetero-forward0"):
            vr = T.axis.spatial(r, v_vr)
            T.reads(I_indptr[0: r + 1], I_indices[0: nnz_i], J_indptr[0: nnz_i + 1], J_indices[0: nnz_j],
                    W_data[0: r * feat_size * feat_size], X_data[0: n * feat_size], Y_data[0: n * r * feat_size])
            T.writes(Y_data[0: n * r * feat_size])
            T.block_attr({"sparse": True})
            W_data_shared = T.alloc_buffer([feat_size * feat_size], dtype="float32", scope="shared")
            for ax0 in T.serial(feat_size * feat_size):
                with T.block("W_data_shared"):
                    v0 = T.axis.spatial(feat_size * feat_size, ax0)
                    T.reads(W_data[feat_size * feat_size * vr + v0])
                    T.writes(W_data_shared[v0])
                    W_data_shared[v0] = W_data[vr * feat_size * feat_size + v0]
            for v_vi, v_vout in T.grid(I_indptr[vr + 1] - I_indptr[vr], feat_size):
                with T.block("rgcn-hetero-forward1"):
                    vi, vout = T.axis.remap("SS", [v_vi, v_vout])
                    T.reads(I_indptr[0: r + 1], I_indices[0: nnz_i], J_indptr[0: nnz_i + 1], J_indices[0: nnz_j],
                            W_data_shared[0: feat_size * feat_size], X_data[0: n * feat_size], Y_data[0: n * r * feat_size])
                    T.writes(Y_data[0: n * r * feat_size])
                    T.block_attr({"sparse": True})
                    for v_vj, v_vin in T.grid(J_indptr[I_indptr[vr] + vi + 1] - J_indptr[I_indptr[vr] + vi], feat_size):
                        with T.block("rgcn-hetero-forward2"):
                            vj, vin = T.axis.remap("RR", [v_vj, v_vin])
                            T.reads(I_indptr[0: r + 1], I_indices[0: nnz_i], J_indptr[0: nnz_i + 1], J_indices[0: nnz_j],
                                    W_data_shared[0: feat_size * feat_size], X_data[0: n * feat_size], Y_data[0: n * r * feat_size])
                            T.writes(Y_data[0: n * r * feat_size])
                            T.block_attr({"sparse": True})
                            with T.init():
                                Y_data[((I_indices[I_indptr[vr] + vi])
                                        * r + vr) * feat_size + vout] = T.float32(0)
                            Y_data[((I_indices[I_indptr[vr] + vi]) * r + vr) * feat_size + vout] = Y_data[((I_indices[I_indptr[vr] + vi]) * r + vr)
                                                                                                          * feat_size + vout] + W_data_shared[vout * feat_size + vin] * X_data[J_indices[J_indptr[I_indptr[vr] + vi] + vj] * feat_size + vin]


def test_lower_rgcn_hetero(g: dgl.DGLHeteroGraph, feat_size: int):
    mod = tvm.IRModule.from_expr(func)
    N, R, FEAT_SIZE, NNZ_I, NNZ_J = mod["main"].params[-5:]
    n = g.num_nodes()
    r = len(g.etypes)
    nnz_j = g.num_edges()

    feat = th.rand(n, feat_size).to(0) / 100
    out = th.zeros(n, r, feat_size).to(0) / 100
    weight = th.rand(r, feat_size, feat_size).to(0)
    W = tvm.nd.array(weight.view(-1).cpu().numpy().astype("float32"), device=tvm.cuda(0))
    X = tvm.nd.array(feat.view(-1).cpu().numpy().astype("float32"), device=tvm.cuda(0))
    Y = tvm.nd.array(out.view(-1).cpu().numpy().astype("float32"), device=tvm.cuda(0))

    indptr_i = [th.LongTensor([0])]
    indices_i = []
    indptr_j = [th.LongTensor([0])]
    indices_j = []
    for etype in g.canonical_etypes:
        src_type, _, dst_type = etype
        etype_id = g.get_etype_id(etype)
        src_type_id = g.get_ntype_id(src_type)
        dst_type_id = g.get_ntype_id(dst_type)
        g_sub = g[etype]
        indptr, indices, _ = g_sub.adj_sparse(fmt="csc")

        unique_nodes = th.nonzero(indptr[:-1] != indptr[1:]).squeeze(1)
        indptr_i.append(th.LongTensor([len(unique_nodes)]))
        indices_i.append(unique_nodes + g.ntype_pointer[dst_type_id])
        indptr_j.append(indptr[unique_nodes] + g.etype_pointer[etype_id])
        indices_j.append(indices + g.ntype_pointer[src_type_id])

    indptr_i = tvm.nd.array(th.cat(indptr_i).numpy().astype("int32"), device=tvm.cuda(0))
    indices_i = tvm.nd.array(th.cat(indices_i).numpy().astype("int32"), device=tvm.cuda(0))
    indptr_j = tvm.nd.array(th.cat(indptr_j).numpy().astype("int32"), device=tvm.cuda(0))
    indices_j = tvm.nd.array(th.cat(indices_j).numpy().astype("int32"), device=tvm.cuda(0))

    nnz_i = indices_i.shape[0]

    sch = tir.Schedule(
        mod["main"].specialize(
            {N: n, R: r, FEAT_SIZE: feat_size, NNZ_I: nnz_i, NNZ_J: nnz_j}
        )
    )

    blk0 = sch.get_block("rgcn-hetero-forward0")
    blk1 = sch.get_block("rgcn-hetero-forward1")
    blk2 = sch.get_block("rgcn-hetero-forward2")
    r, = sch.get_loops(blk0)
    i, f_out = sch.get_loops(blk1)
    j, f_in = sch.get_loops(blk2)
    i1, i2 = sch.split(i, [None, 8])
    sch.bind(i2, "blockIdx.x")
    sch.bind(r, "blockIdx.y")
    sch.bind(f_out, "threadIdx.y")
    sch.bind(f_in, "threadIdx.x")
    f = tvm.build(sch.mod["main"], target="cuda")

    cold_start = 3
    total = 10
    accum = 0

    for epoch in range(10):
        with TorchOpTimer() as timer:
            f(W, X, Y, indptr_i, indices_i, indptr_j, indices_j)
        if epoch >= cold_start:
            accum += timer.time

    print("sparse-tir:\t\t {}ms".format(accum / (total - cold_start)))


if __name__ == "__main__":
    for feat_size in [32]:  # [4, 8, 16, 32, 64]:
        for name in ['bgs']:  # ['aifb', 'mutag', 'bgs', 'am']:
            print('dataset {}:'.format(name))
            dataset = get_dataset_by_name(name)
            g = dataset[0]
            type_pointers = prepare_hetero_graph_simplified(g)
            g.ntype_pointer = type_pointers['ntype_node_pointer']
            g.etype_pointer = type_pointers['etype_edge_pointer']
            test_lower_rgcn_hetero(g, feat_size)
