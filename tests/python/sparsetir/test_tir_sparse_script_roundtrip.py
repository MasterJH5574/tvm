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
import tvm.tir as tir
import tvm.te as te
from tvm.script import tir as T


@T.prim_func
def csrmm(a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle) -> None:
    n = T.var("int32")
    m = T.var("int32")
    k = T.var("int32")
    nnz = T.var("int32")
    I = T.dense_fixed(n)
    J = T.sparse_variable((m, n + 1, nnz), (indptr, indices), "int32")
    K = T.dense_fixed(k)
    A = T.match_sparse_buffer(a, (I, J), nnz, "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), K), m * k, "float32")
    C = T.match_sparse_buffer(c, (I, K), n * k, "float32")
    with T.iter([I, J, K], "SRS", "csrmm") as [vi, vj, vk]:
        with T.init():
            C[vi, vk] = 0.0
        C[vi, vk] = C[vi, vk] + A[vi, vj] * B[vj, vk]


@T.prim_func
def csr_reduce(a: T.handle, b: T.handle, indptr: T.handle, indices: T.handle) -> None:
    n = T.var("int32")
    m = T.var("int32")
    nnz = T.var("int32")
    I = T.dense_fixed(n)
    J = T.sparse_variable((m, n + 1, nnz), (indptr, indices), "int32")
    A = T.match_sparse_buffer(a, (I, J), nnz, "float32")
    B = T.match_sparse_buffer(b, (I,), n, "float32")
    with T.iter([I, J], "SR", "csr_reduce") as [vi, vj]:
        with T.init():
            B[vi] = 0.0
        B[vi] = B[vi] + A[vi, vj]


@T.prim_func
def bsrmm(a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle) -> None:
    nb = T.var("int32")
    mb = T.var("int32")
    nnzb = T.var("int32")
    blk = T.var("int32")
    feat_size = T.var("int32")
    I = T.dense_fixed(nb)
    J = T.sparse_variable((mb, nb + 1, nnzb), (indptr, indices), "int32")
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    F = T.dense_fixed(feat_size)
    A = T.match_sparse_buffer(a, (I, J, BI, BJ), nnzb * blk * blk, "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), BJ, F), mb * blk * feat_size, "float32")
    C = T.match_sparse_buffer(c, (I, BI, F), nb * blk * feat_size, "float32")

    with T.iter([I, J, BI, BJ, F], "SRSSS", "bsrmm") as [
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
def ellpack_mm(a: T.handle, b: T.handle, c: T.handle, indices: T.handle) -> None:
    nb = T.var("int32")
    mb = T.var("int32")
    feat_size = T.var("int32")
    nnz = T.var("int32")
    col = T.var("int32")
    blk = T.var("int32")
    I = T.dense_fixed(nb)
    J = T.sparse_fixed((mb, nnz, col), indices, "int32")
    F = T.dense_fixed(feat_size)
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    A = T.match_sparse_buffer(a, (I, J, BI, BJ), nnz * blk * blk, "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), BJ, F), mb * blk * feat_size, "float32")
    C = T.match_sparse_buffer(c, (I, BI, F), nb * blk * feat_size, "float32")

    with T.iter([I, J, BI, BJ, F], "SRSSS", "bsrmm") as [
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
def csr_element_wise(a: T.handle, b: T.handle, indptr: T.handle, indices: T.handle):
    m = T.var("int32")
    n = T.var("int32")
    nnz = T.var("int32")
    I = T.dense_fixed(m)
    J = T.sparse_variable((n, m + 1, nnz), (indptr, indices), "int32")
    A = T.match_sparse_buffer(a, (I, J), nnz, "float32")
    B = T.match_sparse_buffer(b, (I, J), nnz, "float32")

    with T.iter([I, J], "SS", "csr_element_wise") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0 + 1.0


def test_csrmm():
    func = csrmm
    rt_func = tvm.script.from_source(func.script(show_meta=True))
    tvm.ir.assert_structural_equal(func, rt_func, True)


def test_csr_reduce():
    func = csr_reduce
    rt_func = tvm.script.from_source(func.script(show_meta=True))
    tvm.ir.assert_structural_equal(func, rt_func, True)


def test_bsrmm():
    func = bsrmm
    rt_func = tvm.script.from_source(func.script(show_meta=True))
    tvm.ir.assert_structural_equal(func, rt_func, True)


def test_ellpack_mm():
    func = ellpack_mm
    rt_func = tvm.script.from_source(func.script(show_meta=True))
    tvm.ir.assert_structural_equal(func, rt_func, True)


def test_csr_element_wise():
    func = csr_element_wise
    rt_func = tvm.script.from_source(func.script(show_meta=True))
    tvm.ir.assert_structural_equal(func, rt_func, True)


if __name__ == "__main__":
    test_csrmm()
    test_csr_reduce()
    test_bsrmm()
    test_ellpack_mm()
    test_csr_element_wise()
