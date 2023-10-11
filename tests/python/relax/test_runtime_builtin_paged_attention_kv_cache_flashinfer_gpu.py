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
import scipy.special
import tvm
import tvm.testing
from tvm import dlight as dl
from tvm.contrib import utils
from tvm.script import tir as T


reserved_nseq = 2
total_seq_len = 128
page_size = 16
nlayer = 4
nhead = 16
nfeat = 64
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
# fmt: on


def build_tir_func(tir_funcs: List[tvm.tir.PrimFunc], target="llvm"):
    builts = []
    for tir_func in tir_funcs:
        mod = tvm.IRModule({"main": tir_func})
        with target:
            mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
        f = tvm.build(mod["main"], target=target)
        builts.append(f.entry_func)
    return builts


def get_attention_kernel():
    ext_mod = tvm.runtime.load_static_library(
        "/home/ruihangl/flashinfer/build/CMakeFiles/tvm_binding.dir/src/tvm_wrapper.cu.o",
        ["FlashInferAttentionWithPagedKVCache"],
    )
    assert ext_mod.implements_function("FlashInferAttentionWithPagedKVCache")
    assert ext_mod.is_dso_exportable
    temp_dir = utils.tempdir()
    mod_dso_path = temp_dir.relpath("mod.so")
    ext_mod.export_library(mod_dso_path)
    ext_mod = tvm.runtime.load_module(mod_dso_path)
    assert ext_mod.implements_function("FlashInferAttentionWithPagedKVCache")
    return ext_mod.get_function("FlashInferAttentionWithPagedKVCache")


def f_apply_rotary(x, offset, scale, theta):
    # x: (N, H, F)
    assert len(x.shape) == 3
    nfeat = x.shape[-1]
    nfeat_half = x.shape[-1] // 2
    x = x.astype("float32")
    y = np.concatenate([-x[:, :, nfeat_half:], x[:, :, :nfeat_half]], axis=-1)

    inv_freq = scale / (theta ** (np.arange(0, nfeat, 2).astype("float32") / nfeat))
    t = np.arange(offset, offset + x.shape[0], dtype=inv_freq.dtype)
    freqs = np.einsum("i,j->ij", t, inv_freq)
    emb = np.concatenate((freqs, freqs), axis=-1)
    cos_values = np.cos(emb)
    sin_values = np.sin(emb)

    return np.einsum("ij,ikj->ikj", cos_values, x) + np.einsum("ij,ikj->ikj", sin_values, y)


def test_paged_attention_kv_cache_attention_decode():
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    fprepare = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_prepare")
    fappend = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_append")
    fattention = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_attention")
    (f_transpose_append,) = build_tir_func([transpose_append], target=target)

    attention_kernel = get_attention_kernel()

    device = tvm.cuda()
    cache = fcreate(
        tvm.runtime.ShapeTuple([reserved_nseq, total_seq_len, page_size]),
        nlayer,
        nhead,
        nfeat,
        tvm.nd.empty((), dtype, device=device),
    )

    cached_values = []
    initial_lengths = [6, 8, 11, 19, 43, 21, 24, 35]
    nseq = len(initial_lengths)

    # Initial prefill
    append_lengths = tvm.runtime.ShapeTuple(tuple(length for length in initial_lengths))
    fprepare(cache, append_lengths)

    global_new_kv = np.zeros((nlayer, 2, 0, nhead, nfeat), dtype)
    for length in initial_lengths:
        new_kv = np.random.uniform(-1, 1, size=(nlayer, 2, length, nhead, nfeat)).astype(dtype)
        cached_values.append(new_kv)
        global_new_kv = np.concatenate([global_new_kv, new_kv], axis=2)
    for layer_id in range(nlayer):
        keys = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 0], axis=0), device)
        values = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 1], axis=0), device)
        fappend(cache, f_transpose_append, keys, values, layer_id)

    # Decode
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

    # Attention
    q = np.random.uniform(-1, 1, size=(nlayer, nseq, 1, nhead, nfeat)).astype(dtype)
    for layer_id in range(nlayer):
        q_on_layer = q[layer_id]
        output = tvm.nd.empty((nseq, 1, nhead, nfeat), dtype, device=device)
        fattention(cache, attention_kernel, tvm.nd.array(q_on_layer, device), layer_id, output)

        results = []
        for seq_id in range(nseq):
            q_seq = q_on_layer[seq_id].transpose(1, 0, 2)
            k_seq = cached_values[seq_id][layer_id, 0].transpose(1, 2, 0)
            v_seq = cached_values[seq_id][layer_id, 1].transpose(1, 0, 2)

            results.append(
                np.expand_dims(
                    (
                        scipy.special.softmax(
                            (q_seq.astype("float32") @ k_seq.astype("float32")) / np.sqrt(nfeat),
                            axis=-1,
                        )
                        @ v_seq.astype("float32")
                    ).transpose(1, 0, 2),
                    axis=0,
                ).astype(dtype)
            )
        results = np.concatenate(results, axis=0)
        output = output.numpy()
        tvm.testing.assert_allclose(output, results, atol=1e-3, rtol=1e-3)


def test_paged_attention_kv_cache_attention_decode_with_rope():
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    fprepare = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_prepare")
    fappend = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_append")
    fattention = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_attention")
    (f_transpose_append,) = build_tir_func([transpose_append], target=target)

    attention_kernel = get_attention_kernel()

    device = tvm.cuda()
    cache = fcreate(
        tvm.runtime.ShapeTuple([reserved_nseq, total_seq_len, page_size]),
        nlayer,
        nhead,
        nfeat,
        tvm.nd.empty((), dtype, device=device),
    )

    apply_rotary = True
    rotary_scale = 1.0
    rotary_theta = 1e4

    cached_values = []
    initial_lengths = [6, 8, 11, 19, 43, 21, 24, 35]
    nseq = len(initial_lengths)

    # Initial prefill
    append_lengths = tvm.runtime.ShapeTuple(tuple(length for length in initial_lengths))
    fprepare(cache, append_lengths)

    global_new_kv = np.zeros((nlayer, 2, 0, nhead, nfeat), dtype)
    for length in initial_lengths:
        new_kv = np.random.uniform(-1, 1, size=(nlayer, 2, length, nhead, nfeat)).astype(dtype)
        cached_values.append(new_kv)
        global_new_kv = np.concatenate([global_new_kv, new_kv], axis=2)
    for layer_id in range(nlayer):
        keys = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 0], axis=0), device)
        values = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 1], axis=0), device)
        fappend(cache, f_transpose_append, keys, values, layer_id)

    # Decode
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

    # Attention
    q = np.random.uniform(-1, 1, size=(nlayer, nseq, 1, nhead, nfeat)).astype(dtype)
    for layer_id in range(nlayer):
        q_on_layer = q[layer_id]
        output = tvm.nd.empty((nseq, 1, nhead, nfeat), dtype, device=device)
        fattention(
            cache,
            attention_kernel,
            tvm.nd.array(q_on_layer, device),
            layer_id,
            apply_rotary,
            rotary_scale,
            rotary_theta,
            output,
        )

        results = []
        for seq_id in range(nseq):
            q_seq = f_apply_rotary(
                q_on_layer[seq_id], cached_values[seq_id].shape[2] - 1, rotary_scale, rotary_theta
            ).transpose(1, 0, 2)
            k_seq = f_apply_rotary(
                cached_values[seq_id][layer_id, 0], 0, rotary_scale, rotary_theta
            ).transpose(1, 2, 0)
            v_seq = cached_values[seq_id][layer_id, 1].transpose(1, 0, 2)

            results.append(
                np.expand_dims(
                    (
                        scipy.special.softmax(
                            (q_seq.astype("float32") @ k_seq.astype("float32")) / np.sqrt(nfeat),
                            axis=-1,
                        )
                        @ v_seq.astype("float32")
                    ).transpose(1, 0, 2),
                    axis=0,
                ).astype(dtype)
            )
        results = np.concatenate(results, axis=0)
        output = output.numpy()
        tvm.testing.assert_allclose(output, results, rtol=1e-3, atol=1e-3)


def test_paged_attention_kv_cache_attention_prefill_with_rope():
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    fprepare = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_prepare")
    fappend = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_append")
    fattention = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_attention")
    (f_transpose_append,) = build_tir_func([transpose_append], target=target)
    attention_kernel = get_attention_kernel()

    device = tvm.cuda()
    cache = fcreate(
        tvm.runtime.ShapeTuple([reserved_nseq, total_seq_len, page_size]),
        nlayer,
        nhead,
        nfeat,
        tvm.nd.empty((), dtype, device=device),
    )

    apply_rotary = True
    rotary_scale = 1.0
    rotary_theta = 1e4

    operation_seq = [(0, 6), (1, 8), (2, 11), (3, 19)]
    operation_seq += [(0, 43), (1, 21), (2, 24), (3, 35)]

    current_nseq = 0
    append_lengths_list = []
    cached_values = []

    # Prefill and attention
    for seq_id, append_length in operation_seq:
        if seq_id >= current_nseq:
            assert seq_id == current_nseq
            current_nseq += 1
            cached_values.append(np.zeros((nlayer, 2, 0, nhead, nfeat), dtype))

        append_lengths_list = [0] * current_nseq
        append_lengths_list[seq_id] = append_length

        append_lengths = tvm.runtime.ShapeTuple(append_lengths_list)
        fprepare(cache, append_lengths)

        new_kv = np.random.rand(nlayer, 2, append_length, nhead, nfeat).astype(dtype)
        new_kv[:, 0, ...] = 0.0
        new_kv[:, 1, ...] = 0.0
        # new_kv[:, 1, ...] = 1.0 * append_length
        cached_values[seq_id] = np.concatenate([cached_values[seq_id], new_kv], axis=2)
        for layer_id in range(nlayer):
            keys = tvm.nd.array(np.expand_dims(new_kv[layer_id, 0], axis=0), device)
            values = tvm.nd.array(np.expand_dims(new_kv[layer_id, 1], axis=0), device)
            fappend(cache, f_transpose_append, keys, values, layer_id)

        # q = np.random.uniform(-1, 1, size=(nlayer, 1, append_length, nhead, nfeat)).astype(dtype)
        q = np.zeros((nlayer, 1, append_length, nhead, nfeat)).astype(dtype)
        for layer_id in range(nlayer):
            print(f"layer = {layer_id}")
            q_on_layer = q[layer_id]
            output = tvm.nd.empty((1, append_length, nhead, nfeat), dtype, device=device)
            print(f"q shape = {q_on_layer.shape}, output shape = {output.shape}")
            fattention(
                cache,
                attention_kernel,
                tvm.nd.array(q_on_layer, device),
                layer_id,
                apply_rotary,
                rotary_scale,
                rotary_theta,
                output,
            )

            print(output.numpy())
            exit(0)
            # assert cached_values[seq_id].shape[2] >= append_length
            # rope_offset = cached_values[seq_id].shape[2] - append_length
            # q_seq = f_apply_rotary(
            #     q_on_layer[0],
            #     rope_offset,
            #     rotary_scale,
            #     rotary_theta,
            # ).transpose(1, 0, 2)
            # k_seq = f_apply_rotary(
            #     cached_values[seq_id][layer_id, 0], 0, rotary_scale, rotary_theta
            # ).transpose(1, 2, 0)
            # v_seq = cached_values[seq_id][layer_id, 1].transpose(1, 0, 2)

            # results = np.expand_dims(
            #     (
            #         scipy.special.softmax(
            #             (q_seq.astype("float32") @ k_seq.astype("float32")) / np.sqrt(nfeat),
            #             axis=-1,
            #         )
            #         @ v_seq.astype("float32")
            #     ).transpose(1, 0, 2),
            #     axis=0,
            # ).astype(dtype)

            # output = output.numpy()
            # tvm.testing.assert_allclose(output, results, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    # test_paged_attention_kv_cache_attention_decode()
    # test_paged_attention_kv_cache_attention_decode_with_rope()
    test_paged_attention_kv_cache_attention_prefill_with_rope()
