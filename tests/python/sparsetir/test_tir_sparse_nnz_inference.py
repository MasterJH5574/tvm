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
import scipy.sparse as sp
import numpy as np
from tvm.script import tir as T
from tvm.tir.sparse import AxisTree

@T.prim_func
def csr2bsr_nnz_inf(
    indptr: T.handle, indices: T.handle,
    new_cord: T.handle, glb_counter: T.handle,
    n: T.int32, m: T.int32, nnz: T.int32,
    max_nnz: T.int32) -> None:
    I = T.dense_fixed(n)
    J = T.sparse_variable((m, n + 1, nnz), (indptr, indices), "int32")
    K = T.dense_fixed(2)
    Glb_counter = T.match_buffer(glb_counter, (1,), "int32")
    New_cord = T.match_sparse_buffer(new_cord, (I, J, K), nnz * 2, "int32")
    with T.iter([T.pos(I), T.cord(J), ], "SS", "csr2bsr_nnz_inf") as [vi, vj]:
        #offset = T.atomic_add(Glb_counter.data, 1)
        New_cord[vi, vj, 0] = 0
        New_cord[vi, vj, 1] = 1
        

@T.prim_func
def csr2bsr(indptr_1: T.handle, indices_1: T.handle, indptr_2: T.handle, indices_2: T.handle,
    a_csr: T.handle, a_bsr: T.handle,
    block_size: T.int32,
    n: T.int32, m: T.int32, nnz: T.int32,
    nb: T.int32, mb: T.int32, nnzb: T.int32) -> None:
    I = T.dense_fixed(n)
    J = T.sparse_variable((m, n + 1, nnz), (indptr_1, indices_1), "int32")
    Ibo = T.dense_fixed(nb)
    Jbo = T.sparse_variable((mb, nb + 1, nnzb), (indptr_2, indices_2), "int32")
    Ibi = T.dense_fixed(block_size)
    Jbi = T.dense_fixed(block_size)
    A_csr = T.match_sparse_buffer(a_csr, (I, J), nnz, "float32")
    A_bsr = T.match_sparse_buffer(a_bsr, (Ibo, Jbo, Ibi, Jbi), nnzb * block_size * block_size, "float32")
    with T.iter([T.pos(I), T.cord(J)], "SS", "csr2bsrm") as [vi, vj]:
        A_bsr[T.floordiv(vi, block_size), T.floordiv(vj, block_size), T.floormod(vi, block_size), T.floormod(vj, block_size)] =\
            A_csr[vi, vj]


def test_csr2bsr():
    mod = tvm.IRModule.from_expr(csr2bsr)
    t = AxisTree({
        "J": "I",
        "I": None,
        "K": None,
        "Ibo": None,
        "Jbo": "Ibo",
        "Ibi": None,
        "Ibo": None,
    })
    mod = tvm.tir.transform.LowerSparseTIR(t)(mod)
    print(mod['main'].script())


if __name__ == "__main__":
    test_csr2bsr()