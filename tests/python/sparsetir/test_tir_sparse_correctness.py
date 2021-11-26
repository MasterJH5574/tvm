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
from ctypes import c_float
import tvm
import tvm.testing
from tvm.runtime.ndarray import device
import tvm.tir as tir
import scipy.sparse as sp
import numpy as np
from tvm.script import tir as T


@T.prim_func
def csrmm(a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle, m: T.int32, n: T.int32, k: T.int32, nnz: T.int32) -> None:
    I = T.dense_fixed(m)
    J = T.sparse_variable((n, m + 1, nnz), (indptr, indices), "int32")
    K = T.dense_fixed(k)
    A = T.match_sparse_buffer(a, (I, J), nnz, "float32")
    B = T.match_sparse_buffer(b, (T.to_dense(J), K), n * k, "float32")
    C = T.match_sparse_buffer(c, (I, K), m * k, "float32")
    with T.iter([T.cord(I), T.cord(J), T.cord(K)], "SRS", "csrmm") as [vi, vj, vk]:
        T.block_attr({"sparse": True})
        with T.init():
            C[vi, vk] = 0.0
        C[vi, vk] = C[vi, vk] + A[vi, vj] * B[vj, vk]


@T.prim_func
def csrmm_tir(a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle, M: T.int32, N: T.int32, K: T.int32, NNZ: T.int32) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A_data = T.match_buffer(a, (NNZ,), "float32")
    B = T.match_buffer(b, (N * K,), "float32")
    C = T.match_buffer(c, (M * K,), "float32")
    A_indptr = T.match_buffer(indptr, (M + 1,), "int32")
    A_indices = T.match_buffer(indices, (NNZ,), "int32")
    for i, k in T.grid(M, K):
        with T.block("spmm_outer"):
            vi, vk = T.axis.remap("SS", [i, k])
            with T.init():
                C[vi * K + vk] = 0.
            for j in T.serial(0, A_indptr[vi + 1] - A_indptr[vi]):
                with T.block("spmm_inner"):
                    T.block_attr({"sparse": True})
                    vj = T.axis.R(NNZ, j + A_indptr[vi])
                    C[vi * K + vk] = C[vi * K + vk] + \
                        A_data[vj] * B[A_indices[vj] * K + vk]


@T.prim_func
def bsrmm_tir(a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle, MB: T.int32, NB: T.int32, K: T.int32, BLOCK_SIZE: T.int32, NNZB: T.int32) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A_data = T.match_buffer(a, (NNZB * BLOCK_SIZE * BLOCK_SIZE), "float32")
    B = T.match_buffer(b, (NB * BLOCK_SIZE * K,), "float32")
    C = T.match_buffer(c, (MB * BLOCK_SIZE * K,), "float32")
    A_indptr = T.match_buffer(indptr, (MB + 1,), "int32")
    A_indices = T.match_buffer(indices, (NNZB,), "int32")
    for io, ii, ji, k in T.grid(MB, BLOCK_SIZE, BLOCK_SIZE, K):
        with T.block("spmm_outer"):
            vio, vii, vji, vk = T.axis.remap("SSSS", [io, ii, ji, k])
            with T.init():
                C[(vio * BLOCK_SIZE + vii) * K + vk] = 0.
            for jo in T.serial(0, A_indptr[vio + 1] - A_indptr[vio]):
                with T.block("spmm_inner"):
                    T.block_attr({"sparse": True})
                    vjo = T.axis.R(NNZB, jo + A_indptr[vio])
                    C[(vio * BLOCK_SIZE + vii) * K + vk] = C[(vio * BLOCK_SIZE + vii) * K + vk] + A_data[(
                        vjo * BLOCK_SIZE + vii) * BLOCK_SIZE + vji] * B[(A_indices[vjo] * BLOCK_SIZE + vji) * K + vk]


@T.prim_func
def ellmm_tir(a: T.handle, b: T.handle, c: T.handle, indices: T.handle, M: T.int32, N: T.int32, K: T.int32, NNZ_COLS: T.int32) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A_data = T.match_buffer(a, (M * NNZ_COLS,), "float32")
    B = T.match_buffer(b, (N * K,), "float32")
    C = T.match_buffer(c, (M * K,), "float32")
    A_indices = T.match_buffer(indices, (M * NNZ_COLS,), "int32")
    for i, j, k in T.grid(M, NNZ_COLS, K):
        with T.block("spmm"):
            T.block_attr({"sparse": True})
            vi, vj, vk = T.axis.remap("SRS", [i, j, k])
            with T.init():
                C[vi * K + vk] = 0.
            C[vi * K + vk] = C[vi * K + vk] + A_data[vi * NNZ_COLS + vj] * \
                B[A_indices[vi * NNZ_COLS + vj] * K + vk]


@T.prim_func
def sddmm_tir(a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle, M: T.int32, N: T.int32, K: T.int32, NNZ: T.int32) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalis": True})
    A = T.match_buffer(a, (M * K,), "float32")
    B = T.match_buffer(b, (N * K,), "float32")
    C_data = T.match_buffer(c, (NNZ,), "float32")
    C_indptr = T.match_buffer(indptr, (M + 1,), "int32")
    C_indices = T.match_buffer(indices, (NNZ,), "int32")
    for ij, k in T.grid(NNZ, K):
        with T.block("sddmm"):
            T.block_attr({"sparse": True})
            vij, vk = T.axis.remap("SR", [ij, k])
            T.reads([A[0: M * K], B[0: N * K], C_data[vij], C_indices[vij], C_indptr[0: M + 1]])
            T.writes([C_data[vij]])
            with T.init():
                C_data[vij] = 0.
            C_data[vij] = C_data[vij] + \
                A[(T.upper_bound(C_indptr.data, vij, 0, M + 1) - 1) * K + vk] * B[C_indices[vij] * K + vk]


@T.prim_func
def bmm_tir(a: T.handle, b: T.handle, c: T.handle,
            indptr_i: T.handle, indptr_j: T.handle, indptr_k: T.handle,
            indptr_ij: T.handle, indptr_jk: T.handle, indptr_ik: T.handle,
            BATCH: T.int32,
            NNZ_IJ: T.int32, NNZ_JK: T.int32, NNZ_IK: T.int32) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A = T.match_buffer(a, (NNZ_IJ,), "float32")
    B = T.match_buffer(b, (NNZ_JK,), "float32")
    C = T.match_buffer(c, (NNZ_IK,), "float32")
    indptr_I = T.match_buffer(indptr_i, (BATCH + 1,), "int32")
    indptr_J = T.match_buffer(indptr_j, (BATCH + 1,), "int32")
    indptr_K = T.match_buffer(indptr_k, (BATCH + 1,), "int32")
    indptr_IJ = T.match_buffer(indptr_ij, (BATCH + 1,), "int32")
    indptr_JK = T.match_buffer(indptr_jk, (BATCH + 1,), "int32")
    indptr_IK = T.match_buffer(indptr_ik, (BATCH + 1,), "int32")
    for b in T.grid(BATCH):
        with T.block("bmm_outer"):
            T.block_attr({"sparse": True})
            vb = T.axis.S(BATCH, b)
            with T.init():
                T.evaluate(1)
            for i, j, k in T.grid(indptr_I[vb + 1] - indptr_I[vb], indptr_J[vb + 1] - indptr_J[vb], indptr_K[vb + 1] - indptr_K[vb]):
                with T.block("bmm_inner"):
                    T.block_attr({"sparse": True})
                    vi, vj, vk = T.axis.remap("SRS", [i, j, k])
                    with T.init():
                        C[indptr_IK[vb] + vi * (indptr_K[vb + 1] - indptr_K[vb]) + vk] = 0.
                    C[indptr_IK[vb] + vi * (indptr_K[vb + 1] - indptr_K[vb]) + vk] = C[indptr_IK[vb] + vi * (indptr_K[vb + 1] - indptr_K[vb]) + vk] +\
                        A[indptr_IJ[vb] + vi * (indptr_J[vb + 1] - indptr_J[vb]) + vj] * \
                        B[indptr_JK[vb] + vj * (indptr_K[vb + 1] - indptr_K[vb]) + vk]


def test_csrmm():
    # generate random input
    m = 4096
    n = 4096
    k = 256
    A = sp.random(m, n, dtype="float32", density=0.0125, format='csr')
    nnz = A.nnz
    x = np.random.rand(n, k).astype("float32")
    y_ground_truth = A * x
    y = np.zeros((m * k,)).astype("float32")

    # specialize function
    _, _, _, _, _, M, N, K, NNZ = csrmm_tir.params
    sch = tir.Schedule(
        csrmm_tir.specialize(
            {M: m, N: n, K: k, NNZ: nnz}
        )
    )
    blk_outer = sch.get_block("spmm_outer")
    i, k = sch.get_loops(blk_outer)
    sch.bind(i, "blockIdx.x")
    sch.bind(k, "threadIdx.x")

    # convert numpy tensor to tvm ndarray
    A_indptr = tvm.nd.array(A.indptr.astype("int32"), device=tvm.cuda(0))
    A_indices = tvm.nd.array(A.indices.astype("int32"), device=tvm.cuda(0))
    A_data = tvm.nd.array(A.data.astype("float32"), device=tvm.cuda(0))
    X_nd = tvm.nd.array(x.reshape(-1), device=tvm.cuda(0))
    Y_nd = tvm.nd.array(y, device=tvm.cuda(0))

    # build function
    f = tvm.build(sch.mod, target='cuda')
    f(A_data, X_nd, Y_nd, A_indptr, A_indices)

    # assertion
    tvm.testing.assert_allclose(y_ground_truth.reshape(-1), Y_nd.numpy(), rtol=1e-5)


def test_bsrmm():
    # generate random input
    block_size = 1
    mb = 64
    nb = 64
    k = 256
    m = mb * block_size
    n = nb * block_size
    A_block = sp.random(mb, nb, dtype="float32", density=0.05, format='csr')
    indptr = A_block.indptr
    indices = A_block.indices
    nnzb = A_block.nnz
    data = np.random.rand(nnzb, block_size, block_size)
    A = sp.bsr_matrix((data, indices, indptr), shape=(m, n))
    x = np.random.rand(n, k).astype("float32")
    y_ground_truth = A * x
    y = np.zeros((m * k,)).astype("float32")

    # specialize function
    _, _, _, _, _, MB, NB, K, BLOCK_SIZE, NNZB = bsrmm_tir.params
    sch = tir.Schedule(
        bsrmm_tir.specialize(
            {MB: mb, NB: nb, K: k, BLOCK_SIZE: block_size, NNZB: nnzb}
        )
    )
    blk_outer = sch.get_block("spmm_outer")
    io, ii, ji, k = sch.get_loops(blk_outer)
    sch.unroll(ii)
    sch.unroll(ji)
    sch.bind(io, "blockIdx.x")
    sch.bind(k, "threadIdx.x")

    # convert numpy tensor to tvm ndarray
    A_indptr = tvm.nd.array(indptr.astype("int32"), device=tvm.cuda(0))
    A_indices = tvm.nd.array(indices.astype("int32"), device=tvm.cuda(0))
    A_data = tvm.nd.array(
        data.reshape(-1).astype("float32"), device=tvm.cuda(0))
    X_nd = tvm.nd.array(x.reshape(-1), device=tvm.cuda(0))
    Y_nd = tvm.nd.array(y, device=tvm.cuda(0))

    # build function
    f = tvm.build(sch.mod, target="cuda")
    f(A_data, X_nd, Y_nd, A_indptr, A_indices)

    # assertion
    tvm.testing.assert_allclose(y_ground_truth.reshape(-1), Y_nd.numpy(), rtol=1e-5)


def test_ellmm():
    # generate random input
    nnz_cols = 64
    m = 4096
    n = 4096
    k = 256
    nnz = nnz_cols * m
    indptr = np.arange(0, (m + 1) * nnz_cols, nnz_cols)
    indices = np.random.randint(0, n, size=(nnz,))
    data = np.random.rand(nnz)
    A = sp.csr_matrix((data, indices, indptr), shape=(m, n))
    x = np.random.rand(n, k).astype("float32")
    y_ground_truth = A * x
    y = np.zeros((m * k,)).astype("float32")
    # specialize function
    _, _, _, _, M, N, K, NNZ_COLS = ellmm_tir.params
    sch = tir.Schedule(
        ellmm_tir.specialize(
            {M: m, N: n, K: k, NNZ_COLS: nnz_cols}
        )
    )
    blk = sch.get_block("spmm")
    i, j, k = sch.get_loops(blk)
    sch.bind(i, "blockIdx.x")
    sch.bind(k, "threadIdx.x")
    sch.unroll(j)

    # convert numpy tensor to tvm ndarray
    A_indices = tvm.nd.array(indices.astype("int32"), device=tvm.cuda(0))
    A_data = tvm.nd.array(data.astype("float32"), device=tvm.cuda(0))
    X_nd = tvm.nd.array(x.reshape(-1), device=tvm.cuda(0))
    Y_nd = tvm.nd.array(y, device=tvm.cuda(0))

    # build function
    f = tvm.build(sch.mod, target="cuda")
    f(A_data, X_nd, Y_nd, A_indices)

    # assertion
    tvm.testing.assert_allclose(y_ground_truth.reshape(-1), Y_nd.numpy(), rtol=1e-5)


def test_sddmm():
    # generate random input
    m = 4096
    n = 4096
    k = 256
    C = sp.random(m, n, dtype="float32", density=0.0125, format='csr')
    indptr = C.indptr
    indices = C.indices
    C_coo = C.tocoo()
    nnz = C.nnz
    x = np.random.rand(m, k).astype("float32")
    y = np.random.rand(n, k).astype("float32")
    z_ground_truth = np.matmul(x, y.transpose())[C_coo.row, C_coo.col]
    z = np.zeros((nnz,)).astype("float32")

    # specialize function
    _, _, _, _, _, M, N, K, NNZ = sddmm_tir.params
    sch = tir.Schedule(
        sddmm_tir.specialize(
            {M: m, N: n, K: k, NNZ: nnz}
        )
    )
    blk = sch.get_block("sddmm")
    ij, k = sch.get_loops(blk)
    sch.bind(ij, "blockIdx.x")
    sch.bind(k, "threadIdx.x")

    # convert numpy tensor to tvm ndarray
    C_indices = tvm.nd.array(indices.astype("int32"), device=tvm.cuda(0))
    C_indptr = tvm.nd.array(indptr.astype("int32"), device=tvm.cuda(0))
    X_nd = tvm.nd.array(x.reshape(-1), device=tvm.cuda(0))
    Y_nd = tvm.nd.array(y.reshape(-1), device=tvm.cuda(0))
    C_data = tvm.nd.array(z, device=tvm.cuda(0))

    # build function
    f = tvm.build(sch.mod['main'], target="cuda")
    f(X_nd, Y_nd, C_data, C_indptr, C_indices)

    # assertion
    tvm.testing.assert_allclose(z_ground_truth, C_data.numpy(), rtol=1e-5)


def test_bmm():
    # generate random input
    batch_size = 32
    n_arr = np.random.randint(128, 1024, size=(batch_size,)).astype("int32")
    m_arr = np.random.randint(128, 1024, size=(batch_size,)).astype("int32")
    k_arr = np.random.randint(128, 1024, size=(batch_size,)).astype("int32")
    nm_arr = n_arr * m_arr
    mk_arr = m_arr * k_arr
    nk_arr = n_arr * k_arr
    indptr_n = np.concatenate(([0], n_arr)).cumsum()
    indptr_m = np.concatenate(([0], m_arr)).cumsum()
    indptr_k = np.concatenate(([0], k_arr)).cumsum()
    indptr_nm = np.concatenate(([0], nm_arr)).cumsum()
    indptr_mk = np.concatenate(([0], mk_arr)).cumsum()
    indptr_nk = np.concatenate(([0], nk_arr)).cumsum()
    nnz_ij = indptr_nm[-1]
    nnz_jk = indptr_mk[-1]
    nnz_ik = indptr_nk[-1]
    As = [
        np.random.rand(n, m).astype("float32") for n, m in zip(n_arr, m_arr)
    ]
    Bs = [
        np.random.rand(m, k).astype("float32") for m, k in zip(m_arr, k_arr)
    ]
    Cs = [
        np.matmul(A, B) for A, B in zip(As, Bs)
    ]
    A_flatten = np.concatenate([A.flatten() for A in As], 0)
    B_flatten = np.concatenate([B.flatten() for B in Bs], 0)
    c_flatten = np.concatenate([C.flatten() for C in Cs], 0)

    # specialize function
    _, _, _, _, _, _, _, _, _, BATCH, NNZ_IJ, NNZ_JK, NNZ_IK = bmm_tir.params
    sch = tir.Schedule(
        bmm_tir.specialize({
            BATCH: batch_size, NNZ_IJ: nnz_ij, NNZ_JK: nnz_jk, NNZ_IK: nnz_ik
        })
    )
    bmm_outer = sch.get_block("bmm_outer")
    b, = sch.get_loops(bmm_outer)
    bmm_inner = sch.get_block("bmm_inner")
    i, j, k = sch.get_loops(bmm_inner)
    sch.reorder(i, k, j)
    io, ii = sch.split(i, [None, 32])
    ko, ki = sch.split(k, [None, 32])
    sch.bind(b, "blockIdx.x")
    sch.bind(ki, "threadIdx.x")
    sch.bind(ii, "threadIdx.y")
    sch.decompose_reduction(bmm_inner, j)

    # convert numpy tensor to tvm ndarray
    dev = tvm.cuda(0)
    A_nd = tvm.nd.array(A_flatten, device=dev)
    B_nd = tvm.nd.array(B_flatten, device=dev)
    C_nd = tvm.nd.array(np.zeros_like(c_flatten), device=dev)
    indptr_n_nd = tvm.nd.array(indptr_n.astype("int32"), device=dev)
    indptr_m_nd = tvm.nd.array(indptr_m.astype("int32"), device=dev)
    indptr_k_nd = tvm.nd.array(indptr_k.astype("int32"), device=dev)
    indptr_nm_nd = tvm.nd.array(indptr_nm.astype("int32"), device=dev)
    indptr_mk_nd = tvm.nd.array(indptr_mk.astype("int32"), device=dev)
    indptr_nk_nd = tvm.nd.array(indptr_nk.astype("int32"), device=dev)

    # build function
    f = tvm.build(sch.mod["main"], target="cuda")
    print(f.imported_modules[0].get_source())
    f(A_nd, B_nd, C_nd, indptr_n_nd, indptr_m_nd, indptr_k_nd, indptr_nm_nd, indptr_mk_nd, indptr_nk_nd)

    # assertion
    tvm.testing.assert_allclose(C_nd.numpy(), c_flatten, rtol=1e-5)


if __name__ == "__main__":
    test_csrmm()
    test_bsrmm()
    test_ellmm()
    test_sddmm()
    test_bmm()
