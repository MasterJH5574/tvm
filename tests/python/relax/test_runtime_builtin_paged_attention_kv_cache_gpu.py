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
from typing import List

import numpy as np
import tvm
import tvm.testing
from tvm import dlight as dl
from tvm.script import tir as T


reserved_nseq = 2
total_seq_len = 128
page_size = 8
nlayer = 4
nhead = 16
nfeat = 32
dtype = "float16"

target = tvm.target.Target("nvidia/geforce-rtx-3090-ti")


# fmt: off
@T.prim_func
def transpose_append(
    var_pages: T.handle,
    var_k_data: T.handle,
    var_v_data: T.handle,
    var_page_table_indptr: T.handle,
    var_page_table_values: T.handle,
    var_last_page_offset: T.handle,
    var_append_length_indptr: T.handle,
    var_pos2seqidx: T.handle,
    layer_id: T.int32,
):
    nseq = T.int32()
    ntoken = T.int32()
    nhead = T.int32()
    nfeat = T.int32()
    nlayer = T.int32()
    npage = T.int32()
    page_size = T.int32()
    num_page_chunks = T.int32()
    page_chunk_size = T.int32()

    pages = T.match_buffer(var_pages, (num_page_chunks, nlayer, page_chunk_size, 2, nhead, page_size, nfeat), "float16")
    k_data = T.match_buffer(var_k_data, (ntoken, nhead, nfeat), "float16")
    v_data = T.match_buffer(var_v_data, (ntoken, nhead, nfeat), "float16")
    last_page_offset = T.match_buffer(var_last_page_offset, (nseq,), "int32")
    page_table_indptr = T.match_buffer(var_page_table_indptr, (nseq + 1,), "int32")
    page_table_values = T.match_buffer(var_page_table_values, (npage,), "int32")
    append_length_indptr = T.match_buffer(var_append_length_indptr, (nseq + 1,), "int32")
    pos2seqidx = T.match_buffer(var_pos2seqidx, (ntoken,), "int32")

    for global_pos, h, f in T.grid(ntoken, nhead, nfeat):
        with T.block("k_transpose_append"):
            vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
            seq_idx = pos2seqidx[vgpos]
            seqlen: T.int32 = (page_table_indptr[seq_idx + 1] - page_table_indptr[seq_idx] - 1) * page_size + last_page_offset[seq_idx]
            pages[
                T.floordiv(page_table_values[page_table_indptr[seq_idx] + T.floordiv(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size)], page_chunk_size),
                layer_id,
                T.floormod(page_table_values[page_table_indptr[seq_idx] + T.floordiv(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size)], page_chunk_size),
                0,
                vh,
                T.floormod(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size),
                vf,
            ] = k_data[vgpos, vh, vf]
        with T.block("v_transpose_append"):
            vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
            seq_idx = pos2seqidx[vgpos]
            seqlen: T.int32 = (page_table_indptr[seq_idx + 1] - page_table_indptr[seq_idx] - 1) * page_size + last_page_offset[seq_idx]
            pages[
                T.floordiv(page_table_values[page_table_indptr[seq_idx] + T.floordiv(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size)], page_chunk_size),
                layer_id,
                T.floormod(page_table_values[page_table_indptr[seq_idx] + T.floordiv(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size)], page_chunk_size),
                1,
                vh,
                T.floormod(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size),
                vf,
            ] = v_data[vgpos, vh, vf]


@T.prim_func
def view_cache(
    var_pages: T.handle,
    var_page_table_indptr: T.handle,
    var_page_table_values: T.handle,
    var_values: T.handle,
    seq_id: T.int32,
):
    nhead = T.int32()
    nfeat = T.int32()
    nlayer = T.int32()
    seqlen = T.int32()
    npage = T.int32()
    page_size = T.int32()
    num_page_chunks = T.int32()
    page_chunk_size = T.int32()
    num_total_seqs_plus_1 = T.int32()

    pages = T.match_buffer(var_pages, (num_page_chunks, nlayer, page_chunk_size, 2, nhead, page_size, nfeat), "float16")
    page_table_indptr = T.match_buffer(var_page_table_indptr, (num_total_seqs_plus_1,), "int32")
    page_table_values = T.match_buffer(var_page_table_values, (npage,), "int32")
    values = T.match_buffer(var_values, (nlayer, 2, nhead, seqlen, nfeat), "float16")

    for l, kv_idx, h, pos, f in T.grid(nlayer, 2, nhead, seqlen, nfeat):
        with T.block("view"):
            vl, vi, vh, vp, vf = T.axis.remap("SSSSS", [l, kv_idx, h, pos, f])
            values[vl, vi, vh, vp, vf] = pages[
                T.floordiv(page_table_values[page_table_indptr[seq_id] + T.floordiv(vp, page_size)], page_chunk_size),
                vl,
                T.floormod(page_table_values[page_table_indptr[seq_id] + T.floordiv(vp, page_size)], page_chunk_size),
                vi,
                vh,
                T.floormod(vp, page_size),
                vf,
            ]
# fmt: on


def verify_cached_values(cache, expected, f_view_cache):
    fview = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_view_testing")

    actual = fview(cache, f_view_cache)
    assert len(actual) == len(expected)
    for seq_actual, seq_expected in zip(actual, expected):
        tvm.testing.assert_allclose(np.transpose(seq_actual.numpy(), [0, 1, 3, 2, 4]), seq_expected)


def build_tir_func(tir_funcs: List[tvm.tir.PrimFunc], target="llvm"):
    builts = []
    for tir_func in tir_funcs:
        mod = tvm.IRModule({"main": tir_func})
        with target:
            mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
        f = tvm.build(mod["main"], target=target)
        builts.append(f.entry_func)
    return builts


def test_paged_attention_kv_cache_append_prefill():
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    fprepare = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_prepare")
    fappend = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_append")
    f_transpose_append, f_view_cache = build_tir_func([transpose_append, view_cache], target=target)

    device = tvm.cuda()
    cache = fcreate(
        tvm.runtime.ShapeTuple([reserved_nseq, total_seq_len, page_size]),
        nlayer,
        nhead,
        nfeat,
        tvm.nd.empty((), dtype, device=device),
    )

    operation_seq = [[(0, 6)], [(1, 8)], [(2, 11)], [(3, 16)], [(4, 19), (5, 20)]]
    operation_seq += [[(6, 21), (7, 24)], [(2, 5), (4, 7), (8, 24)]]
    operation_seq += [[(6, 13)], [(8, 19)], [(0, 1)], [(1, 3), (3, 8), (5, 12), (7, 11)]]

    current_nseq = 0
    append_lengths_list = []

    cached_values = []
    for batch in operation_seq:
        for seq_id, _ in batch:
            if seq_id >= current_nseq:
                assert seq_id == current_nseq
                current_nseq += 1

        append_lengths_list = [0] * current_nseq
        for seq_id, append_length in batch:
            append_lengths_list[seq_id] = append_length

        append_lengths = tvm.runtime.ShapeTuple(append_lengths_list)
        print(f"nseq = {current_nseq}")
        print(f"append_lengths = {append_lengths}")
        fprepare(cache, append_lengths)

        global_new_kv = np.zeros((nlayer, 2, 0, nhead, nfeat), dtype)
        for seq_id, new_len in batch:
            if seq_id >= len(cached_values):
                assert seq_id == len(cached_values)
                cached_values.append(np.zeros((nlayer, 2, 0, nhead, nfeat), dtype))

            print(f"seq_id = {seq_id}")
            new_kv = np.random.rand(nlayer, 2, new_len, nhead, nfeat).astype(dtype)
            cached_values[seq_id] = np.concatenate([cached_values[seq_id], new_kv], axis=2)
            global_new_kv = np.concatenate([global_new_kv, new_kv], axis=2)
        for layer_id in range(nlayer):
            print(f"    layer_id = {layer_id}")
            keys = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 0], axis=0), device)
            values = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 1], axis=0), device)
            fappend(cache, f_transpose_append, keys, values, layer_id)

        # Verify
        verify_cached_values(cache, cached_values, f_view_cache)


def test_paged_attention_kv_cache_append_decode():
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    fprepare = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_prepare")
    fappend = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_append")
    f_transpose_append, f_view_cache = build_tir_func([transpose_append, view_cache], target=target)

    device = tvm.cuda()
    cache = fcreate(
        tvm.runtime.ShapeTuple([reserved_nseq, total_seq_len, page_size]),
        nlayer,
        nhead,
        nfeat,
        tvm.nd.empty((), dtype, device=device),
    )

    cached_values = []
    initial_lengths = [31, 21, 16, 3, 8, 7, 3]
    nseq = len(initial_lengths)

    # Initial prefill
    append_lengths = tvm.runtime.ShapeTuple(tuple(length for length in initial_lengths))
    fprepare(cache, append_lengths)

    global_new_kv = np.zeros((nlayer, 2, 0, nhead, nfeat), dtype)
    for length in initial_lengths:
        new_kv = np.random.rand(nlayer, 2, length, nhead, nfeat).astype(dtype)
        cached_values.append(new_kv)
        global_new_kv = np.concatenate([global_new_kv, new_kv], axis=2)
    for layer_id in range(nlayer):
        print(f"    layer_id = {layer_id}")
        keys = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 0], axis=0), device)
        values = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 1], axis=0), device)
        fappend(cache, f_transpose_append, keys, values, layer_id)

    verify_cached_values(cache, cached_values, f_view_cache)

    # Decode
    for _ in range(16):
        decode_new_kv = np.random.rand(nlayer, 2, nseq, 1, nhead, nfeat).astype(dtype)
        fprepare(cache)
        for seq_id in range(nseq):
            cached_values[seq_id] = np.concatenate(
                [cached_values[seq_id], decode_new_kv[:, :, seq_id, ...]], axis=2
            )
        for layer_id in range(nlayer):
            keys = tvm.nd.array(decode_new_kv[layer_id, 0], device)
            values = tvm.nd.array(decode_new_kv[layer_id, 1], device)
            fappend(cache, f_transpose_append, keys, values, layer_id)

        verify_cached_values(cache, cached_values, f_view_cache)


def test_paged_attention_kv_cache_remove():
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    fprepare = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_prepare")
    fappend = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_append")
    fremove = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_remove")
    f_transpose_append, f_view_cache = build_tir_func([transpose_append, view_cache], target=target)

    device = tvm.cuda()
    cache = fcreate(
        tvm.runtime.ShapeTuple([reserved_nseq, total_seq_len, page_size]),
        nlayer,
        nhead,
        nfeat,
        tvm.nd.empty((), dtype, device=device),
    )

    cached_values = []
    initial_lengths = [31, 21, 16, 3, 8, 7, 3]
    nseq = len(initial_lengths)

    # Initial prefill
    append_lengths = tvm.runtime.ShapeTuple(tuple(length for length in initial_lengths))
    fprepare(cache, append_lengths)

    global_new_kv = np.zeros((nlayer, 2, 0, nhead, nfeat), dtype)
    for length in initial_lengths:
        new_kv = np.random.rand(nlayer, 2, length, nhead, nfeat).astype(dtype)
        cached_values.append(new_kv)
        global_new_kv = np.concatenate([global_new_kv, new_kv], axis=2)
    for layer_id in range(nlayer):
        print(f"    layer_id = {layer_id}")
        keys = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 0], axis=0), device)
        values = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 1], axis=0), device)
        fappend(cache, f_transpose_append, keys, values, layer_id)

    verify_cached_values(cache, cached_values, f_view_cache)

    # Remove
    while len(cached_values) > 2:
        seq_id = np.random.randint(0, len(cached_values))
        fremove(cache, seq_id)
        cached_values.pop(seq_id)
        verify_cached_values(cache, cached_values, f_view_cache)

    # Append after removal
    seq_id = 2
    new_len = 29
    fprepare(cache, tvm.runtime.ShapeTuple((0, 0, new_len)))
    new_kv = np.random.rand(nlayer, 2, new_len, nhead, nfeat).astype(dtype)
    cached_values.append(new_kv)
    for layer_id in range(nlayer):
        keys = tvm.nd.array(np.expand_dims(new_kv[layer_id, 0], axis=0), device)
        values = tvm.nd.array(np.expand_dims(new_kv[layer_id, 1], axis=0), device)
        fappend(cache, f_transpose_append, keys, values, layer_id)

    verify_cached_values(cache, cached_values, f_view_cache)


def test_paged_attention_kv_cache_popn():
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    fprepare = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_prepare")
    fappend = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_append")
    fpopn = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_popn")
    f_transpose_append, f_view_cache = build_tir_func([transpose_append, view_cache], target=target)

    device = tvm.cuda()
    cache = fcreate(
        tvm.runtime.ShapeTuple([reserved_nseq, total_seq_len, page_size]),
        nlayer,
        nhead,
        nfeat,
        tvm.nd.empty((), dtype, device=device),
    )

    cached_values = []
    initial_lengths = [20, 24, 26, 27]
    nseq = len(initial_lengths)

    # Initial prefill
    append_lengths = tvm.runtime.ShapeTuple(tuple(length for length in initial_lengths))
    fprepare(cache, append_lengths)

    global_new_kv = np.zeros((nlayer, 2, 0, nhead, nfeat), dtype)
    for length in initial_lengths:
        new_kv = np.random.rand(nlayer, 2, length, nhead, nfeat).astype(dtype)
        cached_values.append(new_kv)
        global_new_kv = np.concatenate([global_new_kv, new_kv], axis=2)
    for layer_id in range(nlayer):
        print(f"    layer_id = {layer_id}")
        keys = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 0], axis=0), device)
        values = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 1], axis=0), device)
        fappend(cache, f_transpose_append, keys, values, layer_id)

    verify_cached_values(cache, cached_values, f_view_cache)

    # Pop n
    for pop_length in [3, 13]:
        for seq_id in range(nseq):
            print(f"pop {pop_length} for seq {seq_id}")
            fpopn(cache, seq_id, pop_length)
            cached_values[seq_id] = cached_values[seq_id][:, :, :-pop_length, ...]
            verify_cached_values(cache, cached_values, f_view_cache)

    # Decode after pop n
    for _ in range(5):
        decode_new_kv = np.random.rand(nlayer, 2, nseq, 1, nhead, nfeat).astype(dtype)
        fprepare(cache)
        for seq_id in range(nseq):
            cached_values[seq_id] = np.concatenate(
                [cached_values[seq_id], decode_new_kv[:, :, seq_id, ...]], axis=2
            )
        for layer_id in range(nlayer):
            keys = tvm.nd.array(decode_new_kv[layer_id, 0], device)
            values = tvm.nd.array(decode_new_kv[layer_id, 1], device)
            fappend(cache, f_transpose_append, keys, values, layer_id)

        verify_cached_values(cache, cached_values, f_view_cache)


def test_paged_attention_kv_cache_clear():
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    fprepare = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_prepare")
    fappend = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_append")
    fclear = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_clear")
    f_transpose_append, f_view_cache = build_tir_func([transpose_append, view_cache], target=target)

    device = tvm.cuda()
    cache = fcreate(
        tvm.runtime.ShapeTuple([reserved_nseq, total_seq_len, page_size]),
        nlayer,
        nhead,
        nfeat,
        tvm.nd.empty((), dtype, device=device),
    )

    cached_values = []
    initial_lengths = [20, 24, 26, 27]

    # Initial prefill
    append_lengths = tvm.runtime.ShapeTuple(tuple(length for length in initial_lengths))
    fprepare(cache, append_lengths)

    global_new_kv = np.zeros((nlayer, 2, 0, nhead, nfeat), dtype)
    for length in initial_lengths:
        new_kv = np.random.rand(nlayer, 2, length, nhead, nfeat).astype(dtype)
        cached_values.append(new_kv)
        global_new_kv = np.concatenate([global_new_kv, new_kv], axis=2)
    for layer_id in range(nlayer):
        print(f"    layer_id = {layer_id}")
        keys = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 0], axis=0), device)
        values = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 1], axis=0), device)
        fappend(cache, f_transpose_append, keys, values, layer_id)

    verify_cached_values(cache, cached_values, f_view_cache)

    # Clear
    fclear(cache)
    verify_cached_values(cache, [], f_view_cache)


if __name__ == "__main__":
    test_paged_attention_kv_cache_append_prefill()
    test_paged_attention_kv_cache_append_decode()
    test_paged_attention_kv_cache_remove()
    test_paged_attention_kv_cache_popn()
    test_paged_attention_kv_cache_clear()
    # tvm.testing.main()
