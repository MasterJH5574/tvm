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
import tvm.tir as tir
import scipy.sparse as sp
import numpy as np
import pytest
from tvm.script import tir as T


@T.prim_func
def csrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    n: T.int32,
    m: T.int32,
    k: T.int32,
    nnz: T.int32,
) -> None:
    I = T.dense_fixed(n)
    J = T.sparse_variable(I, (m, nnz), (indptr, indices), "int32")
    K = T.dense_fixed(k)
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), K), "float32")
    C = T.match_sparse_buffer(c, (I, K), "float32")
    with T.iter([I, K, J], "SSR", "csrmm") as [vi, vk, vj]:
        with T.init():
            C[vi, vk] = 0.0
        C[vi, vk] = C[vi, vk] + A[vi, vj] * B[vj, vk]


@T.prim_func
def csrmm_dense_iter(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    n: T.int32,
    m: T.int32,
    k: T.int32,
    nnz: T.int32,
) -> None:
    I = T.dense_fixed(n)
    J = T.sparse_variable(I, (m, nnz), (indptr, indices), "int32")
    K = T.dense_fixed(k)
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), K), "float32")
    C = T.match_sparse_buffer(c, (I, K), "float32")
    with T.iter([I, T.dense(J), K], "SRS", "csrmm") as [vi, vj, vk]:
        with T.init():
            C[vi, vk] = 0.0
        C[vi, vk] = C[vi, vk] + A[vi, vj] * B[vj, vk]


@T.prim_func
def segment_reduce(
    a: T.handle,
    b: T.handle,
    indptr: T.handle,
    n: T.int32,
    nnz: T.int32,
) -> None:
    I = T.dense_fixed(n)
    J = T.dense_variable(I, (100, nnz), indptr, "int32")
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (I,), "float32")
    with T.iter([I, J], "SR", "segment_reduce") as [vi, vj]:
        with T.init():
            B[vi] = 0.
        B[vi] = B[vi] + A[vi, vj]


@T.prim_func
def lowered_csrmm(a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle, n: T.int32, m: T.int32, k: T.int32, nnz: T.int32) -> None:
    A_data = T.match_buffer(a, [nnz], dtype="float32")
    B_data = T.match_buffer(b, [m * k], dtype="float32")
    C_data = T.match_buffer(c, [n * k], dtype="float32")
    J_indptr = T.match_buffer(indptr, [n + 1], dtype="int32")
    J_indices = T.match_buffer(indices, [nnz], dtype="int32")
    for v_vi, v_vk in T.grid(n, k):
        with T.block("csrmm_outer"):
            vi, vk = T.axis.remap("SS", [v_vi, v_vk])
            T.reads([J_indptr[0: n + 1], J_indices[0: nnz],
                    A_data[0: nnz], B_data[0: m * k], C_data[0: n * k]])
            T.writes([C_data[0: n * k]])
            T.block_attr({"sparse": True})
            for v_vj in T.serial(0, J_indptr[v_vi + 1] - J_indptr[v_vi]):
                with T.block("csrmm"):
                    vj = T.axis.reduce(J_indptr[v_vi + 1] - J_indptr[v_vi], v_vj)
                    T.reads([J_indptr[0: n + 1], J_indices[0: nnz],
                            A_data[0: nnz], B_data[0: m * k], C_data[0: n * k]])
                    T.writes([C_data[0: n * k]])
                    T.block_attr({"sparse": True})
                    with T.init():
                        C_data[vi * k + vk] = T.float32(0)
                    C_data[vi * k + vk] = C_data[vi * k + vk] + A_data[J_indptr[vi] + vj] * \
                        B_data[J_indices[J_indptr[vi] + vj] * k + vk]


@T.prim_func
def csr_reduce(
    a: T.handle,
    b: T.handle,
    indptr: T.handle,
    indices: T.handle,
    n: T.int32,
    m: T.int32,
    nnz: T.int32,
) -> None:
    I = T.dense_fixed(n)
    J = T.sparse_variable(I, (m, nnz), (indptr, indices), "int32")
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (I,), "float32")
    with T.iter([I, J], "SR", "csr_reduce") as [vi, vj]:
        with T.init():
            B[vi] = 0.0
        B[vi] = B[vi] + A[vi, vj]


@T.prim_func
def lowered_csr_reduce(a: T.handle, b: T.handle, indptr: T.handle, indices: T.handle, n: T.int32, m: T.int32, nnz: T.int32) -> None:
    A_data = T.match_buffer(a, [nnz], dtype="float32")
    B_data = T.match_buffer(b, [n], dtype="float32")
    J_indptr = T.match_buffer(indptr, [n + 1], dtype="int32")
    J_indices = T.match_buffer(indices, [nnz], dtype="int32")
    for v_vi in T.serial(0, n):
        with T.block("csr_reduce_outer"):
            vi = T.axis.spatial(n, v_vi)
            T.reads([J_indptr[0: n + 1], J_indices[0: nnz], A_data[0: nnz], B_data[0: n]])
            T.writes([B_data[0: n]])
            T.block_attr({"sparse": True})
            for v_vj in T.serial(0, J_indptr[v_vi + 1] - J_indptr[v_vi]):
                with T.block("csr_reduce"):
                    vj = T.axis.reduce(J_indptr[v_vi + 1] - J_indptr[v_vi], v_vj)
                    T.reads([J_indptr[0: n + 1], J_indices[0: nnz], A_data[0: nnz], B_data[0: n]])
                    T.writes([B_data[0: n]])
                    T.block_attr({"sparse": True})
                    with T.init():
                        B_data[vi] = T.float32(0)
                    B_data[vi] = B_data[vi] + A_data[J_indptr[vi] + vj]


@T.prim_func
def bsrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    nb: T.int32,
    mb: T.int32,
    nnzb: T.int32,
    blk: T.int32,
    feat_size: T.int32,
) -> None:
    I = T.dense_fixed(nb)
    J = T.sparse_variable(I, (mb, nnzb), (indptr, indices), "int32")
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    F = T.dense_fixed(feat_size)
    A = T.match_sparse_buffer(a, (I, J, BI, BJ), "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), BJ, F), "float32")
    C = T.match_sparse_buffer(c, (I, BI, F), "float32")

    with T.iter([I, BI, BJ, F, J], "SSRSR", "bsrmm") as [
        vi,
        vbi,
        vbj,
        vf,
        vj,
    ]:
        with T.init():
            C[vi, vbi, vf] = 0.0
        C[vi, vbi, vf] = C[vi, vbi, vf] + A[vi, vj, vbi, vbj] * B[vj, vbj, vf]


@T.prim_func
def lowered_bsrmm(a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle, nb: T.int32, mb: T.int32, nnzb: T.int32, blk: T.int32, feat_size: T.int32) -> None:
    A_data = T.match_buffer(a, [nnzb * blk * blk], dtype="float32")
    B_data = T.match_buffer(b, [mb * blk * feat_size], dtype="float32")
    C_data = T.match_buffer(c, [nb * blk * feat_size], dtype="float32")
    J_indptr = T.match_buffer(indptr, [nb + 1], dtype="int32")
    J_indices = T.match_buffer(indices, [nnzb], dtype="int32")
    for v_vi, v_vbi, v_vbj, v_vf in T.grid(nb, blk, blk, feat_size):
        with T.block("bsrmm_outer"):
            vi, vbi, vbj, vf = T.axis.remap("SSRS", [v_vi, v_vbi, v_vbj, v_vf])
            T.reads([J_indptr[0: nb + 1], J_indices[0: nnzb], A_data[0: nnzb * blk * blk],
                    B_data[0: mb * blk * feat_size], C_data[0: nb * blk * feat_size]])
            T.writes([C_data[0: nb * blk * feat_size]])
            T.block_attr({"sparse": True})
            with T.init():
                C_data[(vi * blk + vbi) * feat_size + vf] = T.float32(0)
            for v_vj in T.serial(0, J_indptr[v_vi + 1] - J_indptr[v_vi]):
                with T.block("bsrmm"):
                    vj = T.axis.reduce(J_indptr[v_vi + 1] - J_indptr[v_vi], v_vj)
                    T.reads([J_indptr[0: nb + 1], J_indices[0: nnzb], A_data[0: nnzb * blk * blk],
                            B_data[0: mb * blk * feat_size], C_data[0: nb * blk * feat_size]])
                    T.writes([C_data[0: nb * blk * feat_size]])
                    T.block_attr({"sparse": True})
                    C_data[(vi * blk + vbi) * feat_size + vf] = C_data[(vi * blk + vbi) * feat_size + vf] + A_data[(
                        (J_indptr[vi] + vj) * blk + vbi) * blk + vbj] * B_data[(J_indices[J_indptr[vi] + vj] * blk + vbj) * feat_size + vf]


@T.prim_func
def ellpack_mm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indices: T.handle,
    nb: T.int32,
    mb: T.int32,
    feat_size: T.int32,
    col: T.int32,
    blk: T.int32,
) -> None:
    I = T.dense_fixed(nb)
    J = T.sparse_fixed(I, (mb, col), indices, "int32")
    F = T.dense_fixed(feat_size)
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    A = T.match_sparse_buffer(a, (I, J, BI, BJ), "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), BJ, F), "float32")
    C = T.match_sparse_buffer(c, (I, BI, F), "float32")

    with T.iter([I, J, BI, BJ, F], "SRSRS", "ellmm") as [
        vi,
        vj,
        vbi,
        vbj,
        vf,
    ]:
        with T.init():
            C[vi, vbi, vf] = 0.0
        C[vi, vbi, vf] = C[vi, vbi, vf] + A[vi, vj, vbi, vbj] * B[vj, vbj, vf]


@T.prim_func
def lowered_ellpack_mm(a: T.handle, b: T.handle, c: T.handle, indices: T.handle, nb: T.int32, mb: T.int32, feat_size: T.int32, col: T.int32, blk: T.int32) -> None:
    A_data = T.match_buffer(a, [nb * col * blk * blk], dtype="float32")
    B_data = T.match_buffer(b, [mb * blk * feat_size], dtype="float32")
    C_data = T.match_buffer(c, [nb * blk * feat_size], dtype="float32")
    J_indices = T.match_buffer(indices, [nb * col], dtype="int32")
    # body
    # with T.block("root")
    for v_vi, v_vj, v_vbi, v_vbj, v_vf in T.grid(nb, col, blk, blk, feat_size):
        with T.block("ellmm"):
            vi, vj, vbi, vbj, vf = T.axis.remap("SRSRS", [v_vi, v_vj, v_vbi, v_vbj, v_vf])
            T.reads([J_indices[0 : nb * col], A_data[0 : nb * col * blk * blk], B_data[0 : mb * blk * feat_size], C_data[0 : nb * blk * feat_size]])
            T.writes([C_data[0 : nb * blk * feat_size]])
            T.block_attr({"sparse":True})
            with T.init():
                C_data[(vi * blk + vbi) * feat_size + vf] = T.float32(0)
            C_data[(vi * blk + vbi) * feat_size + vf] = C_data[(vi * blk + vbi) * feat_size + vf] + A_data[((vi * col + vj) * blk + vbi) * blk + vbj] * B_data[(J_indices[vi * col + vj] * blk + vbj) * feat_size + vf]


@T.prim_func
def csr_element_wise(
    a: T.handle,
    b: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    nnz: T.int32,
) -> None:
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (I, J), "float32")

    with T.iter([I, J], "SS", "csr_element_wise") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.5


@T.prim_func
def lowered_csr_element_wise(a: T.handle, b: T.handle, indptr: T.handle, indices: T.handle, m: T.int32, n: T.int32, nnz: T.int32) -> None:
    A_data = T.match_buffer(a, [nnz], dtype="float32")
    B_data = T.match_buffer(b, [nnz], dtype="float32")
    J_indptr = T.match_buffer(indptr, [m + 1], dtype="int32")
    J_indices = T.match_buffer(indices, [nnz], dtype="int32")
    for v_vi in T.serial(0, m):
        with T.block("csr_element_wise_outer"):
            vi = T.axis.spatial(m, v_vi)
            T.reads([J_indptr[0: m + 1], J_indices[0: nnz], A_data[0: nnz]])
            T.writes([B_data[0: nnz]])
            T.block_attr({"sparse": True})
            for v_vj in T.serial(0, J_indptr[v_vi + 1] - J_indptr[v_vi]):
                with T.block("csr_element_wise"):
                    vj = T.axis.spatial(J_indptr[v_vi + 1] - J_indptr[v_vi], v_vj)
                    T.reads([J_indptr[0: m + 1], J_indices[0: nnz], A_data[0: nnz]])
                    T.writes([B_data[0: nnz]])
                    T.block_attr({"sparse": True})
                    B_data[J_indptr[vi] + vj] = A_data[J_indptr[vi] + vj] * T.float32(2.5)

@T.prim_func
def bmm(
    x: T.handle,
    y: T.handle,
    z: T.handle,
    indptr_i: T.handle,
    indptr_j: T.handle,
    indptr_k: T.handle,
    indptr_ij: T.handle,
    indptr_jk: T.handle,
    indptr_ik: T.handle,
    batch_size: T.int32,
    nnz_i: T.int32,
    nnz_j: T.int32,
    nnz_k: T.int32,
    nnz_ij: T.int32,
    nnz_jk: T.int32,
    nnz_ik: T.int32
) -> None:
    B = T.dense_fixed(batch_size)
    I = T.dense_variable(B, (32768, nnz_i), indptr_i, "int32")
    J = T.dense_variable(B, (32768, nnz_j), indptr_j, "int32")
    K = T.dense_variable(B, (32768, nnz_k), indptr_k, "int32")
    IJ = T.attach_axis(I, J, nnz_ij, indptr_ij, "int32")
    JK = T.attach_axis(J, K, nnz_jk, indptr_jk, "int32")
    IK = T.attach_axis(I, K, nnz_ik, indptr_ik, "int32")
    X = T.match_sparse_buffer(x, (B, I, IJ), "float32")
    Y = T.match_sparse_buffer(y, (B, J, JK), "float32")
    Z = T.match_sparse_buffer(z, (B, I, IK), "float32")
    with T.iter([B, I, J, K], "SSRS", "bmm") as [vb, vi, vj, vk]:
        with T.init():
            Z[vb, vi, vk] = 0.
        Z[vb, vi, vk] = Z[vb, vi, vk] + X[vb, vi, vk] * Y[vb, vk, vj]


@T.prim_func
def sddmm(a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle, m: T.int32, n: T.int32, k: T.int32, nnz: T.int32) -> None:
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    K = T.dense_fixed(k)
    A = T.match_sparse_buffer(a, (I, K), "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), K), "float32")
    C = T.match_sparse_buffer(c, (I, J), "float32")

    with T.iter([I, J, K], "SSR", "sddmm") as [vi, vj, vk]:
        with T.init():
            C[vi, vj] = 0. 
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def fused_sddmm(a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle, m: T.int32, n: T.int32, k: T.int32, nnz: T.int32) -> None:
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    K = T.dense_fixed(k)
    A = T.match_sparse_buffer(a, (I, K), "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), K), "float32")
    C = T.match_sparse_buffer(c, (I, J), "float32")

    with T.iter([T.fuse(I, J), K], "SSR", "sddmm") as [vi, vj, vk]:
        with T.init():
            C[vi, vj] = 0. 
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def square_sum(a: T.handle, b: T.handle, indptr_j: T.handle, indices_j: T.handle, indptr_k: T.handle, indices_k: T.handle, nnz_j: T.int32, nnz_k: T.int32, M: T.int32, N1: T.int32, N2: T.int32):
    I = T.dense_fixed(M)
    J = T.sparse_variable(I, (N1, nnz_j), (indptr_j, indices_j), "int32")
    K = T.sparse_variable(J, (N2, nnz_k), (indptr_k, indices_k), "int32")
    A = T.match_sparse_buffer(a, (I, J, K), "float32")
    B = T.match_sparse_buffer(b, (I,), "float32")

    with T.iter([I, J, K], "SRR", "square_sum") as [vi, vj, vk]:
        with T.init():
            B[vi] = 0.0
        B[vi] = B[vi] + A[vi, vj, vk]


@T.prim_func
def lowered_square_sum(a: T.handle, b: T.handle, indptr_j: T.handle, indices_j: T.handle, indptr_k: T.handle, indices_k: T.handle, nnz_j: T.int32, nnz_k: T.int32, M: T.int32, N1: T.int32, N2: T.int32) -> None:
    A_data = T.match_buffer(a, [nnz_k], dtype="float32")
    B_data = T.match_buffer(b, [M], dtype="float32")
    J_indptr = T.match_buffer(indptr_j, [M + 1], dtype="int32")
    J_indices = T.match_buffer(indices_j, [nnz_j], dtype="int32")
    K_indptr = T.match_buffer(indptr_k, [nnz_j + 1], dtype="int32")
    K_indices = T.match_buffer(indices_k, [nnz_k], dtype="int32")

    for v_vi in T.serial(0, M):
        with T.block("square_sum_2"):
            vi = T.axis.spatial(M, v_vi)
            T.reads([J_indptr[0 : M + 1], J_indices[0 : nnz_j], K_indptr[0 : nnz_j + 1], K_indices[0 : nnz_k], A_data[0 : nnz_k], B_data[0 : M]])
            T.writes([B_data[0 : M]])
            T.block_attr({"sparse":True})
            for v_vj in T.serial(0, J_indptr[v_vi + 1] - J_indptr[v_vi]):
                with T.block("square_sum_1"):
                    vj = T.axis.reduce(J_indptr[v_vi + 1] - J_indptr[v_vi], v_vj)
                    T.reads([J_indptr[0 : M + 1], J_indices[0 : nnz_j], K_indptr[0 : nnz_j + 1], K_indices[0 : nnz_k], A_data[0 : nnz_k], B_data[0 : M]])
                    T.writes([B_data[0 : M]])
                    T.block_attr({"sparse":True})
                    with T.init():
                        B_data[vi] = T.float32(0)
                    for v_vk in T.serial(0, K_indptr[J_indptr[v_vi] + v_vj + 1] - K_indptr[J_indptr[v_vi] + v_vj]):
                        with T.block("square_sum"):
                            vk = T.axis.reduce(K_indptr[J_indptr[v_vi] + v_vj + 1] - K_indptr[J_indptr[v_vi] + v_vj], v_vk)
                            T.reads([J_indptr[0 : M + 1], J_indices[0 : nnz_j], K_indptr[0 : nnz_j + 1], K_indices[0 : nnz_k], A_data[0 : nnz_k], B_data[0 : M]])
                            T.writes([B_data[0 : M]])
                            T.block_attr({"sparse":True})
                            B_data[vi] = B_data[vi] + A_data[K_indptr[J_indptr[vi] + vj] + vk]


@T.prim_func
def square_sum_two_K(a: T.handle, b: T.handle, indptr_j: T.handle, indices_j: T.handle, indptr_k0: T.handle, indices_k0: T.handle, indptr_k1: T.handle, indices_k1: T.handle, nnz_j: T.int32, nnz_k: T.int32, M: T.int32, N1: T.int32, N2: T.int32):
    # Used only for testing `GetIndicesRange()`.
    # Currently it is ensured that `indptr_k0` is the same as `indptr_k1`, and `indices_k0` is the
    # same as `indices_k1`.
    I = T.dense_fixed(M)
    J = T.sparse_variable(I, (N1, nnz_j), (indptr_j, indices_j), "int32")
    K0 = T.sparse_variable(J, (N2, nnz_k), (indptr_k0, indices_k0), "int32")
    K1 = T.sparse_variable(J, (N2, nnz_k), (indptr_k1, indices_k1), "int32")
    A = T.match_sparse_buffer(a, (I, J, K0), "float32")
    B = T.match_sparse_buffer(b, (I,), "float32")

    with T.iter([I, J, K1], "SRR", "square_sum") as [vi, vj, vk]:
        with T.init():
            B[vi] = 0.0
        B[vi] = B[vi] + A[vi, vj, vk]


@T.prim_func
def lowered_square_sum_two_K(a: T.handle, b: T.handle, indptr_j: T.handle, indices_j: T.handle, indptr_k0: T.handle, indices_k0: T.handle, indptr_k1: T.handle, indices_k1: T.handle, nnz_j: T.int32, nnz_k: T.int32, M: T.int32, N1: T.int32, N2: T.int32) -> None:
    A_data = T.match_buffer(a, [nnz_k], dtype="float32")
    B_data = T.match_buffer(b, [M], dtype="float32")
    J_indptr = T.match_buffer(indptr_j, [M + 1], dtype="int32")
    J_indices = T.match_buffer(indices_j, [nnz_j], dtype="int32")
    K0_indptr = T.match_buffer(indptr_k0, [nnz_j + 1], dtype="int32")
    K0_indices = T.match_buffer(indices_k0, [nnz_k], dtype="int32")
    K1_indptr = T.match_buffer(indptr_k1, [nnz_j + 1], dtype="int32")
    K1_indices = T.match_buffer(indices_k1, [nnz_k], dtype="int32")

    for v_vi in T.serial(0, M):
        with T.block("square_sum_2"):
            vi = T.axis.spatial(M, v_vi)
            T.reads([J_indptr[0 : M + 1], J_indices[0 : nnz_j], K0_indptr[0 : nnz_j + 1], K0_indices[0 : nnz_k], K1_indptr[0 : nnz_j + 1], K1_indices[0 : nnz_k], A_data[0 : nnz_k], B_data[0 : M]])
            T.writes([B_data[0 : M]])
            T.block_attr({"sparse":True})
            for v_vj in T.serial(0, J_indptr[v_vi + 1] - J_indptr[v_vi]):
                with T.block("square_sum_1"):
                    vj = T.axis.reduce(J_indptr[v_vi + 1] - J_indptr[v_vi], v_vj)
                    T.reads([J_indptr[0 : M + 1], J_indices[0 : nnz_j], K0_indptr[0 : nnz_j + 1], K0_indices[0 : nnz_k], K1_indptr[0 : nnz_j + 1], K1_indices[0 : nnz_k], A_data[0 : nnz_k], B_data[0 : M]])
                    T.writes([B_data[0 : M]])
                    T.block_attr({"sparse":True})
                    with T.init():
                        B_data[vi] = T.float32(0)
                    for v_vk in T.serial(0, K1_indptr[J_indptr[v_vi] + v_vj + 1] - K1_indptr[J_indptr[v_vi] + v_vj]):
                        with T.block("square_sum"):
                            vk = T.axis.reduce(K1_indptr[J_indptr[v_vi] + v_vj + 1] - K1_indptr[J_indptr[v_vi] + v_vj], v_vk)
                            T.reads([J_indptr[0 : M + 1], J_indices[0 : nnz_j], K0_indptr[0 : nnz_j + 1], K0_indices[0 : nnz_k], K1_indptr[0 : nnz_j + 1], K1_indices[0 : nnz_k], A_data[0 : nnz_k], B_data[0 : M]])
                            T.writes([B_data[0 : M]])
                            T.block_attr({"sparse":True})
                            B_data[vi] = B_data[vi] + A_data[T.tvm_lower_bound(K0_indices.data, K1_indices[K1_indptr[J_indptr[vi] + vj] + vk], K0_indptr[J_indptr[vi] + vj], K0_indptr[J_indptr[vi] + vj + 1], dtype="int32")]


def test_csrmm():
    mod = tvm.IRModule.from_expr(csrmm)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_csrmm, True)

    A = sp.random(512, 512, dtype="float32", density=0.0125, format="csr")
    x = np.random.rand(512, 128).astype("float32")
    y_ground_truth = A * x
    y = np.zeros((512, 128)).astype("float32")

    n, m, k, nnz = mod["main"].params[-4:]
    f = tvm.build(mod["main"].specialize({n: 512, m: 512, k: 128, nnz: A.nnz}), target="llvm")

    ctx = tvm.cpu(0)
    A_indptr = tvm.nd.array(A.indptr.astype("int32"), device=ctx)
    A_indices = tvm.nd.array(A.indices.astype("int32"), device=ctx)
    A_data = tvm.nd.array(A.data.astype("float32"), device=ctx)
    X_nd = tvm.nd.array(x.reshape(-1), device=ctx)
    Y_nd = tvm.nd.array(y.reshape(-1), device=ctx)
    f(A_data, X_nd, Y_nd, A_indptr, A_indices)
    tvm.testing.assert_allclose(y_ground_truth.reshape(-1), Y_nd.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.skip(reason="Under implementation")
def test_csrmm_dense_iter():
    mod = tvm.IRModule.from_expr(csrmm_dense_iter)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    # tvm.ir.assert_structural_equal(mod["main"], lowered_csrmm, True)
    # Todo


@pytest.mark.skip(reason="Under implementation")
def test_segment_reduce():
    mod = tvm.IRModule.from_expr(segment_reduce)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    # Todo


def test_csr_reduce():
    mod = tvm.IRModule.from_expr(csr_reduce)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_csr_reduce, True)

    A = sp.random(128, 128, dtype="float32", density=0.0125, format="csr")
    b_ground_truth = np.array(np.sum(A, axis=1))
    b = np.zeros((128,)).astype("float32")

    n, m, nnz = csr_reduce.params[-3:]
    f = tvm.build(mod["main"].specialize({n: 128, m: 128, nnz: A.nnz}), target="llvm")

    ctx = tvm.cpu(0)
    A_indptr = tvm.nd.array(A.indptr.astype("int32"), device=ctx)
    A_indices = tvm.nd.array(A.indices.astype("int32"), device=ctx)
    A_data = tvm.nd.array(A.data.astype("float32"), device=ctx)
    B_nd = tvm.nd.array(b, device=ctx)
    f(A_data, B_nd, A_indptr, A_indices)
    tvm.testing.assert_allclose(b_ground_truth.reshape(-1), B_nd.numpy(), rtol=1e-5, atol=1e-5)


def test_bsrmm():
    mod = tvm.IRModule.from_expr(bsrmm)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_bsrmm, True)

    block_size = 16
    nb = 32
    mb = 32
    feat_size = 256
    n = nb * block_size
    m = mb * block_size

    A_block = sp.random(mb, nb, dtype="float32", density=0.05, format="csr")
    indptr = A_block.indptr
    indices = A_block.indices
    nnzb = A_block.nnz
    data = np.random.rand(nnzb, block_size, block_size)
    A = sp.bsr_matrix((data, indices, indptr), shape=(n, m))
    x = np.random.rand(m, feat_size).astype("float32")
    y_ground_truth = A * x
    y = np.zeros((n * feat_size,)).astype("float32")

    v_nb, v_mb, v_nnzb, v_blk, v_feat_size = bsrmm.params[-5:]
    f = tvm.build(
        mod["main"].specialize(
            {v_nb: nb, v_mb: mb, v_nnzb: nnzb, v_blk: block_size, v_feat_size: feat_size}
        ),
        target="llvm",
    )

    ctx = tvm.cpu(0)
    A_indptr = tvm.nd.array(indptr.astype("int32"), device=ctx)
    A_indices = tvm.nd.array(indices.astype("int32"), device=ctx)
    A_data = tvm.nd.array(data.reshape(-1).astype("float32"), device=ctx)
    X_nd = tvm.nd.array(x.reshape(-1), device=ctx)
    Y_nd = tvm.nd.array(y, device=ctx)
    f(A_data, X_nd, Y_nd, A_indptr, A_indices)
    tvm.testing.assert_allclose(y_ground_truth.reshape(-1), Y_nd.numpy(), rtol=1e-5, atol=1e-5)


def test_ellpack_mm():
    mod = tvm.IRModule.from_expr(ellpack_mm)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_ellpack_mm, True)

    nnz_cols = 4
    nb = 64
    mb = 64
    feat_size = 1024
    nnz = nb * nnz_cols
    block_size = 16
    n = nb * block_size
    m = mb * block_size

    rng = np.random.default_rng()
    indptr = np.arange(0, (nb + 1) * nnz_cols, nnz_cols)
    indices = np.array([rng.choice(mb, size=nnz_cols, replace=False) for i in range(nb)])
    order = indices.argsort(axis=1)
    indices = np.array([indices[i, order[i]] for i in range(0, nb)]).reshape(-1)
    data = np.random.rand(nnz, block_size, block_size)
    A = sp.bsr_matrix((data, indices, indptr), shape=(n, m))
    x = np.random.rand(m, feat_size).astype("float32")
    y_ground_truth = A * x
    y = np.zeros((n * feat_size,)).astype("float32")

    v_nb, v_mb, v_feat_size, v_col, v_blk = ellpack_mm.params[-5:]
    f = tvm.build(
        mod["main"].specialize(
            {
                v_nb: nb,
                v_mb: mb,
                v_feat_size: feat_size,
                v_col: nnz_cols,
                v_blk: block_size,
            }
        ),
        target="llvm",
    )

    ctx = tvm.cpu(0)
    A_indices = tvm.nd.array(indices.astype("int32"), device=ctx)
    A_data = tvm.nd.array(data.reshape(-1).astype("float32"), device=ctx)
    X_nd = tvm.nd.array(x.reshape(-1), device=ctx)
    Y_nd = tvm.nd.array(y, device=ctx)
    f(A_data, X_nd, Y_nd, A_indices)
    tvm.testing.assert_allclose(y_ground_truth.reshape(-1), Y_nd.numpy(), rtol=1e-5, atol=1e-5)


def test_csr_element_wise():
    mod = tvm.IRModule.from_expr(csr_element_wise)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_csr_element_wise, True)

    A = sp.random(128, 128, dtype="float32", density=0.0125, format="csr")
    b_ground_truth = A * 2.5
    b = np.zeros((A.nnz,)).astype("float32")

    m, n, nnz = csr_element_wise.params[-3:]
    f = tvm.build(mod["main"].specialize({m: 128, n: 128, nnz: A.nnz}), target="llvm")

    ctx = tvm.cpu(0)
    A_indptr = tvm.nd.array(A.indptr.astype("int32"), device=ctx)
    A_indices = tvm.nd.array(A.indices.astype("int32"), device=ctx)
    A_data = tvm.nd.array(A.data.astype("float32"), device=ctx)
    B_nd = tvm.nd.array(b, device=ctx)
    f(A_data, B_nd, A_indptr, A_indices)
    tvm.testing.assert_allclose(b_ground_truth.data.reshape(-1), B_nd.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.skip(reason="Under implementation")
def test_bmm():
    mod = tvm.IRModule.from_expr(bmm)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    # TODO


def test_sddmm():
    mod = tvm.IRModule.from_expr(sddmm)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    print(mod['main'].script())
    # TODO


def test_fused_sddmm():
    mod = tvm.IRModule.from_expr(fused_sddmm)
    print(mod['main'].script())
    # TODO


def test_square_sum():
    mod = tvm.IRModule.from_expr(square_sum)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_square_sum, True)

    density = 0.0125
    M = N1 = N2 = 128
    A_J = sp.random(M, N1, dtype="float32", density=1 - (1 - density) ** N2, format="csr")
    indptr_j = A_J.indptr
    indices_j = A_J.indices
    nnz_j = A_J.nnz
    A_K = sp.random(nnz_j, N2, dtype="float32", density=density, format="csr")
    indptr_k = A_K.indptr
    indices_k = A_K.indices
    nnz_k = A_K.nnz
    data = A_K.data

    b_ij = np.asarray(A_K.sum(axis=1)).squeeze()
    A_J = sp.csr_matrix((b_ij, indices_j, indptr_j), shape=(M, N1))
    b_ground_truth = np.asarray(A_J.sum(axis=1)).squeeze()
    b = np.zeros((M,)).astype("float32")

    v_nnz_j, v_nnz_k, v_M, v_N1, v_N2 = square_sum.params[-5:]
    f = tvm.build(mod["main"].specialize({v_nnz_j: nnz_j, v_nnz_k: nnz_k, v_M: M, v_N1: N1, v_N2: N2}), target="llvm")

    ctx = tvm.cpu(0)
    A_data = tvm.nd.array(data.astype("float32"), device=ctx)
    A_indptr_j = tvm.nd.array(indptr_j.astype("int32"), device=ctx)
    A_indices_j = tvm.nd.array(indices_j.astype("int32"), device=ctx)
    A_indptr_k = tvm.nd.array(indptr_k.astype("int32"), device=ctx)
    A_indices_k = tvm.nd.array(indices_k.astype("int32"), device=ctx)
    B_data = tvm.nd.array(b.astype("float32"), device=ctx)
    f(A_data, B_data, A_indptr_j, A_indices_j, A_indptr_k, A_indices_k)

    tvm.testing.assert_allclose(b_ground_truth, B_data.numpy(), rtol=1e-5, atol=1e-5)


def test_square_sum_two_K():
    mod = tvm.IRModule.from_expr(square_sum_two_K)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_square_sum_two_K, True)

    sch = tir.Schedule(mod, debug_mask="all")
    i, = sch.get_loops(sch.get_block("square_sum_2"))
    sch.bind(i, "threadIdx.x")

    density = 0.0125
    M = N1 = N2 = 128
    A_J = sp.random(M, N1, dtype="float32", density=1 - (1 - density) ** N2, format="csr")
    indptr_j = A_J.indptr
    indices_j = A_J.indices
    nnz_j = A_J.nnz
    A_K = sp.random(nnz_j, N2, dtype="float32", density=density, format="csr")
    indptr_k = A_K.indptr
    indices_k = A_K.indices
    nnz_k = A_K.nnz
    data = A_K.data

    b_ij = np.asarray(A_K.sum(axis=1)).squeeze()
    A_J = sp.csr_matrix((b_ij, indices_j, indptr_j), shape=(M, N1))
    b_ground_truth = np.asarray(A_J.sum(axis=1)).squeeze()
    b = np.zeros((M,)).astype("float32")

    v_nnz_j, v_nnz_k, v_M, v_N1, v_N2 = square_sum_two_K.params[-5:]
    f = tvm.build(sch.mod["main"].specialize({v_nnz_j: nnz_j, v_nnz_k: nnz_k, v_M: M, v_N1: N1, v_N2: N2}), target="cuda")

    ctx = tvm.device("cuda")
    A_data = tvm.nd.array(data.astype("float32"), device=ctx)
    A_indptr_j = tvm.nd.array(indptr_j.astype("int32"), device=ctx)
    A_indices_j = tvm.nd.array(indices_j.astype("int32"), device=ctx)
    A_indptr_k0 = tvm.nd.array(indptr_k.astype("int32"), device=ctx)
    A_indices_k0 = tvm.nd.array(indices_k.astype("int32"), device=ctx)
    A_indptr_k1 = tvm.nd.array(indptr_k.astype("int32"), device=ctx)
    A_indices_k1 = tvm.nd.array(indices_k.astype("int32"), device=ctx)
    B_data = tvm.nd.array(b.astype("float32"), device=ctx)
    f(A_data, B_data, A_indptr_j, A_indices_j, A_indptr_k0, A_indices_k0, A_indptr_k1, A_indices_k1)

    tvm.testing.assert_allclose(b_ground_truth, B_data.numpy(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_csrmm()
    test_csrmm_dense_iter()
    test_segment_reduce()
    test_csr_reduce()
    test_bsrmm()
    test_ellpack_mm()
    test_csr_element_wise()
    test_sddmm()
    test_fused_sddmm()
    test_bmm()
    test_square_sum()
    test_square_sum_two_K()
