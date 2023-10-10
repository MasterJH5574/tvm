/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file src/runtime/relax_vm/lm_support.cc
 * \brief Runtime to support language model related task
 *
 * Including inplace attention kv cache for runtime and simple sampler.
 *
 * This file provides a simple implementation of inplace attention
 * kv cache for relax runtime. The main goal here is to help us enable
 * auto-regressive decoding quickly in relax.
 *
 * This is not the only way to support attention kv-cache.
 * Our support of attention kv-cache can subject to future
 * changes as we build more LM verticals.
 *
 * We will keep the impact minimum by puting it as a private
 * runtime builtin provide as in this file.
 *
 * We can evolve this implementation as we build more LM verticals.
 */
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/relax_vm/memory_manager.h>
#include <tvm/runtime/relax_vm/vm.h>

#include <cmath>

namespace tvm {
namespace runtime {
namespace relax_vm {

//-------------------------------------------
// We keep the implementation private as
// they may subject to future changes.
//
// Users can interact with it through the
// runtime API function calls
//-------------------------------------------
/*!
 * \brief An object representing an attention kv cache.
 */
class AttentionKVCacheObj : public Object {
 public:
  /*!
   * \brief Underlying support data.
   */
  NDArray data;

  /*!
   * \brief number of slots already filled.
   */
  int64_t fill_count{0};

  /*!
   * \brief View all current cached values as one array.
   * \param shape The cached values.
   */
  NDArray View(const ShapeTuple& shape) {
    CHECK_EQ(shape[0], fill_count) << "Requested shape do not match the filled count";
    for (int i = 1; i < this->data->ndim; ++i) {
      CHECK_EQ(shape[i], data->shape[i]) << "Dimension " << i << " mismatch";
    }
    return data.CreateView(shape, data->dtype);
  }

  /** Clear the cache */
  void Clear() { this->fill_count = 0; }

  /** pop n entries */
  void PopN(size_t n) {
    ICHECK_LE(n, fill_count);
    this->fill_count -= n;
  }

  void Update(NDArray value) {
    CHECK(data.DataType() == value.DataType()) << "dtype mismatch";
    CHECK_EQ(value->shape[0], fill_count) << "Requested shape do not match the filled count";
    ICHECK(data.IsContiguous());
    ICHECK(value.IsContiguous());

    DLTensor copy_dst = *(data.operator->());
    copy_dst.byte_offset = 0;
    copy_dst.shape = value->shape;
    NDArray::CopyFromTo(value.operator->(), &copy_dst);
    this->fill_count = value->shape[0];
  }

  /*!
   * \brief Append value to the cache.
   * \param value The value to be appended.
   */
  void Append(NDArray value) {
    CHECK(data.DataType() == value.DataType()) << "dtype mismatch";
    // reallocate cache
    int64_t reserved_slots = data->shape[0];
    while (fill_count + value->shape[0] > reserved_slots) {
      reserved_slots *= 2;
    }
    if (reserved_slots != data->shape[0]) {
      std::vector<int64_t> new_shape(data->shape, data->shape + data->ndim);
      new_shape[0] = reserved_slots;
      NDArray new_data = NDArray::Empty(new_shape, data->dtype, data->device);
      new_data.CreateView(data.Shape(), data->dtype).CopyFrom(data);
      this->data = new_data;
    }
    // copy into the fill count position.
    ICHECK_LE(fill_count + value->shape[0], data->shape[0]);
    ICHECK(data.IsContiguous());

    int64_t num_filled_elements = fill_count;
    for (int i = 1; i < data->ndim; ++i) {
      CHECK_EQ(value->shape[i], data->shape[i]) << "Dimension " << i << " mismatch";
      num_filled_elements *= data->shape[i];
    }
    // create a view of copy dest to copy the value into it.
    DLTensor copy_dst = *(data.operator->());
    copy_dst.byte_offset = num_filled_elements * ((data->dtype.bits * data->dtype.lanes + 7) / 8);
    copy_dst.shape = value->shape;
    NDArray::CopyFromTo(value.operator->(), &copy_dst);
    this->fill_count += value->shape[0];
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.vm.AttentionKVCache";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttentionKVCacheObj, Object);
};

/*! \brief reference to closure. */
class AttentionKVCache : public ObjectRef {
 public:
  /*!
   * \brief Create the attention kv cache.
   * \param init_data The initial reserved.
   */
  static AttentionKVCache Create(NDArray init_data, ShapeTuple reserve_shape, int init_fill_count) {
    auto n = make_object<AttentionKVCacheObj>();
    n->data = NDArray::Empty(reserve_shape, init_data->dtype, init_data->device);
    n->fill_count = 0;
    n->Append(init_data);
    if (init_fill_count >= 0) {
      n->fill_count = init_fill_count;
    }
    return AttentionKVCache(n);
  }

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(AttentionKVCache, ObjectRef, AttentionKVCacheObj);
};

TVM_REGISTER_OBJECT_TYPE(AttentionKVCacheObj);

/*!
 * \brief Create multiple kv caches with same shape, from single memory allocation.
 * \param init_data The initial data to put into the cache. Ignored if init_fill_count is
 *        less than 0.
 * \param reserve_shape The shape of cache.
 * \param init_fill_count The initial row to fill into
 *        the cache.
 * \param num_caches Number of caches to create.
 */
Array<AttentionKVCache> CreateMultipleKVCaches(NDArray init_data, ShapeTuple reserve_shape,
                                               int init_fill_count, int num_caches) {
  DLDataType dtype = init_data->dtype;

  int64_t cache_size = (dtype.bits * dtype.lanes + 7) / 8;
  for (const auto dim : reserve_shape) {
    cache_size *= dim;
  }

  // Add padding to make each cache align to kAllocAlignment
  using tvm::runtime::kAllocAlignment;
  int64_t padding = (kAllocAlignment - cache_size % kAllocAlignment) % kAllocAlignment;
  int64_t cache_offset = cache_size + padding;

  Storage storage =
      Storage(MemoryManager::GetOrCreateAllocator(init_data->device, AllocatorType::kNaive)
                  ->Alloc(cache_offset * num_caches, kAllocAlignment, dtype));

  Array<AttentionKVCache> result;
  for (int i = 0; i < num_caches; ++i) {
    auto c = make_object<AttentionKVCacheObj>();
    c->data = storage->AllocNDArray(i * cache_offset, reserve_shape, dtype);
    c->fill_count = 0;
    if (init_fill_count > 0) {
      c->Append(init_data);
      c->fill_count = init_fill_count;
    }
    result.push_back(AttentionKVCache(c));
  }
  return result;
}

//-------------------------------------------------
//  Register runtime functions
//-------------------------------------------------
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_create")
    .set_body_typed(AttentionKVCache::Create);

TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_create_multiple")
    .set_body_typed(CreateMultipleKVCaches);

AttentionKVCache AttentionKVCacheUpdate(AttentionKVCache cache, NDArray value) {
  cache->Update(value);
  return cache;
}

TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_update").set_body_typed(AttentionKVCacheUpdate);

AttentionKVCache AttentionKVCacheAppend(AttentionKVCache cache, NDArray value) {
  cache->Append(value);
  return cache;
}

TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_append").set_body_typed(AttentionKVCacheAppend);

NDArray AttentionKVCacheView(AttentionKVCache cache, ShapeTuple shape) {
  return cache->View(shape);
}

TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_view")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.size() == 1 || args.size() == 2)
          << "ValueError: `vm.builtin.attention_kv_cache_view` expects 1 or 2 arguments, but got "
          << args.size() << ".";
      AttentionKVCache cache = args[0];
      if (args.size() == 2) {
        ShapeTuple shape = args[1];
        *rv = cache->View(shape);
      } else {
        std::vector<ShapeTuple::index_type> shape;
        shape.push_back(cache->fill_count);
        for (int i = 1; i < cache->data->ndim; ++i) {
          shape.push_back(cache->data->shape[i]);
        }
        *rv = cache->View(ShapeTuple(shape));
      }
    });

void AttentionKVCacheArrayPopN(Array<AttentionKVCache> caches, int64_t n) {
  for (AttentionKVCache cache : caches) {
    cache->PopN(static_cast<size_t>(n));
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_array_popn")
    .set_body_typed(AttentionKVCacheArrayPopN);

void AttentionKVCacheArrayClear(Array<AttentionKVCache> caches) {
  for (AttentionKVCache cache : caches) {
    cache->Clear();
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_array_clear")
    .set_body_typed(AttentionKVCacheArrayClear);

// NOTE this is a built-in highly related to LM so we put it here.
int SampleTopPFromLogits(NDArray logits, double temperature, double top_p, double uniform_sample) {
  ICHECK(logits.IsContiguous());
  ICHECK(logits.DataType() == DataType::Float(32));

  if (logits->device.device_type != kDLCPU) {
    logits = logits.CopyTo(DLDevice{kDLCPU, 0});
  }

  ICHECK(logits->device.device_type == kDLCPU);

  for (int i = 0; i < logits->ndim - 1; ++i) {
    ICHECK_EQ(logits->shape[i], 1) << "The leading dimensions of logits must be 1";
  }

  std::vector<std::pair<float, int>> data;
  data.resize(logits->shape[logits->ndim - 1]);
  const float* plogits = static_cast<float*>(logits->data);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = std::make_pair(plogits[i], static_cast<int>(i));
  }

  auto fcmp = [](const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
  };
  // sort by logits from largest to smallest
  std::sort(data.begin(), data.end(), fcmp);

  // argmax
  if (temperature < 1e-6f) {
    return data[0].second;
  }

  // compute expf scaled by temp
  float sum = 0.0f, logit_scale = 1.0f / temperature;
  float max_value = data[0].first;
  for (auto it = data.begin(); it != data.end(); ++it) {
    it->first = expf((it->first - max_value) * logit_scale);
    sum += it->first;
  }

  // do a cumsum in order of data
  float cum_sum_prob = 0.0f;
  float top_p_sum = 0.0f;
  for (auto it = data.begin(); it != data.end(); ++it) {
    float prob = it->first / sum;
    if (cum_sum_prob < top_p) {
      top_p_sum += prob;
    }
    cum_sum_prob += prob;
    it->first = cum_sum_prob;
  }
  // pick a number based on random in (0, 1)
  for (auto it = data.begin(); it != data.end(); ++it) {
    if (uniform_sample < it->first / top_p_sum) {
      return it->second;
    }
  }
  ICHECK_LE(uniform_sample, data[0].first);
  return data[0].second;
}

TVM_REGISTER_GLOBAL("vm.builtin.sample_top_p_from_logits").set_body_typed(SampleTopPFromLogits);

int SampleTopPFromProb(NDArray prob, double top_p, double uniform_sample) {
  ICHECK(prob.IsContiguous());
  ICHECK(prob.DataType() == DataType::Float(32));

  if (prob->device.device_type != kDLCPU) {
    prob = prob.CopyTo(DLDevice{kDLCPU, 0});
  }

  ICHECK(prob->device.device_type == kDLCPU);

  for (int i = 0; i < prob->ndim - 1; ++i) {
    ICHECK_EQ(prob->shape[i], 1) << "The leading dimensions of logits must be 1";
  }

  // Key observation: when we are doing top_p sampling
  // usually we only need to preserve some of the elements with
  // high probablities before we do sort
  std::vector<std::pair<float, int>> data;
  int64_t ndata = prob->shape[prob->ndim - 1];
  const float* p_prob = static_cast<float*>(prob->data);

  auto sample_top_p_with_filter = [&](float cuttoff) -> int64_t {
    data.clear();
    // filter the data with cuttoff
    for (int64_t i = 0; i < ndata; ++i) {
      if (p_prob[i] >= cuttoff) {
        data.emplace_back(std::make_pair(p_prob[i], static_cast<int>(i)));
      }
    }
    if (data.size() == 0) return -1;
    auto fcmp = [](const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
      return lhs.first > rhs.first;
    };
    std::sort(data.begin(), data.end(), fcmp);

    // short cut, if we know that
    // uniform sample < p[0] / top_p
    // we know that unform_sample < p[0] / top_p_sum
    // because top_p_sum gaurantees to be smaller than top_p
    // so we can simply return the argmax sample
    // without computing anything
    if (uniform_sample < data[0].first / top_p) return data[0].second;

    // compute top_p_sum
    float cum_sum_prob = 0.0f;
    float top_p_sum = 0.0f;
    for (auto it = data.begin(); it != data.end(); ++it) {
      float prob = it->first;
      if (cum_sum_prob < top_p) {
        top_p_sum += prob;
      } else {
        // we get to the right cutoff pt
        break;
      }
      cum_sum_prob += prob;
      it->first = cum_sum_prob;
    }
    // we find that the current total sum by the given cutoff
    // is not sufficient to cover everything
    // this means we might need to retry a smaller cutoff pt.
    if (cum_sum_prob < top_p && cuttoff != 0.0f) return -1;

    for (auto it = data.begin(); it != data.end(); ++it) {
      if (uniform_sample < it->first / top_p_sum) {
        return it->second;
      }
    }
    return data[data.size() - 1].second;
  };

  if (top_p < 1) {
    // sample through cutoff by a number
    // by pigeonhole principle we will get at most 1024 elements
    // usually it is much less by applying this filtering(order of 10 - 20)
    data.reserve(128);
    int64_t sampled_index = sample_top_p_with_filter(top_p / 1024);
    if (sampled_index >= 0) return sampled_index;
  }
  // fallback via full prob, rare case
  data.reserve(ndata);
  int64_t sampled_index = sample_top_p_with_filter(0.0f);
  ICHECK_GE(sampled_index, 0);
  return sampled_index;
}

TVM_REGISTER_GLOBAL("vm.builtin.sample_top_p_from_prob").set_body_typed(SampleTopPFromProb);

// This is an inplace operation.
void ApplyRepetitionPenalty(NDArray logits, NDArray token_ids, double penalty) {
  ICHECK(logits.IsContiguous());
  ICHECK(token_ids.IsContiguous());
  ICHECK(logits.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
  ICHECK(token_ids.DataType() == DataType::Int(32)) << "token ids must be int32!";
  ICHECK(logits->device.device_type == kDLCPU) << "logits device must be CPU!";
  ICHECK(token_ids->device.device_type == kDLCPU) << "token_ids device must be CPU!";
  float* logits_raw_data = static_cast<float*>(logits->data);
  int* token_ids_data = static_cast<int*>(token_ids->data);
  size_t num_token_ids = token_ids->shape[token_ids->ndim - 1];
  for (size_t i = 0; i < num_token_ids; ++i) {
    int token_id = token_ids_data[i];
    if (logits_raw_data[token_id] <= 0) {
      logits_raw_data[token_id] *= penalty;
    } else {  // logits > 0
      logits_raw_data[token_id] /= penalty;
    }
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.apply_repetition_penalty").set_body_typed(ApplyRepetitionPenalty);

// This is an inplace operation.
void ApplySoftmaxWithTemperature(NDArray logits, double temperature) {
  ICHECK(logits.IsContiguous());
  ICHECK(logits.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
  ICHECK(logits->device.device_type == kDLCPU) << "logits device must be CPU!";
  int vocab_size = logits->shape[logits->ndim - 1];
  float* logits_raw_data = static_cast<float*>(logits->data);
  float inv_temp = 1.0f / temperature;
  float m = std::numeric_limits<float>::min();
  double d = 0.0f;
  for (int i = 0; i < vocab_size; ++i) {
    float x = logits_raw_data[i] * inv_temp;
    float m_prev = m;
    m = std::max(m, x);
    d = d * std::exp(m_prev - m) + std::exp(x - m);
  }
  for (int i = 0; i < vocab_size; ++i) {
    float x = logits_raw_data[i] * inv_temp;
    logits_raw_data[i] = std::exp(x - m) / d;
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.apply_softmax_with_temperature")
    .set_body_typed(ApplySoftmaxWithTemperature);

////////////////////////////////////////////////////////////////////////

class PagedAttentionKVCacheObj : public Object {
 public:
  int64_t num_total_seqs;
  int64_t num_pages_in_use;
  int64_t num_pages_allocated;

  int64_t page_size;
  int64_t nlayer;
  int64_t nhead;
  int64_t nfeat;

  const DLDataType dtype_aux = DataType::Int(32, 1).operator DLDataType();

  /********************* Page Structures *********************/

  const int64_t page_chunk_size = 1;

  NDArray pages;
  std::vector<int32_t> free_page_ids;

  std::vector<std::vector<int32_t>> page_table;
  std::vector<int32_t> seqlen;

  NDArray page_table_indptr_device;
  NDArray page_table_values_device;
  NDArray last_page_offset_device;

  /********************* Current Batch Info *********************/

  ShapeTuple cur_append_lengths;
  NDArray cur_append_length_indptr_device;
  NDArray cur_pos2seqidx_device;

 public:
  void Attention(PackedFunc f_attention, NDArray q_data, int64_t layer_id, NDArray output,
                 bool apply_rotary = false, double rotary_scale = 1.0f, double rotary_theta = 1e4) {
    // q_data: (b, len, nhead, nfeat)
    CHECK_EQ(q_data->ndim, 4);
    CHECK_GT(q_data->shape[1], 0);
    CHECK_EQ(q_data->shape[2], nhead);
    CHECK_EQ(q_data->shape[3], nfeat);
    CHECK(q_data.DataType() == pages.DataType());

    int64_t ntoken = 0;
    for (int64_t i = 0; i < num_total_seqs; ++i) {
      ntoken += cur_append_lengths[i];
      if (q_data->shape[0] > 1) {
        CHECK_EQ(cur_append_lengths[i], q_data->shape[1]);
      }
    }
    if (q_data->shape[0] > 1) {
      CHECK_EQ(q_data->shape[0], num_total_seqs);
    }
    CHECK_EQ(ntoken, q_data->shape[0] * q_data->shape[1]);

    // Provide a 8MB size float temp buffer for the attention kernel.
    static NDArray tmp_buffer =
        NDArray::Empty({8 * 1024 * 1024}, DLDataType(DataType::Float(32)), pages->device);

    f_attention(q_data, pages,                                                                //
                page_table_indptr_device.CreateView({num_total_seqs + 1}, dtype_aux),         //
                page_table_values_device.CreateView({num_pages_in_use}, dtype_aux),           //
                last_page_offset_device.CreateView({num_total_seqs}, dtype_aux),              //
                cur_append_length_indptr_device.CreateView({num_total_seqs + 1}, dtype_aux),  //
                layer_id, tmp_buffer, output, apply_rotary, rotary_scale, rotary_theta);
  }

  void Prepare(Optional<ShapeTuple> opt_append_lengths) {
    ShapeTuple append_lengths;
    int64_t nseq =
        opt_append_lengths.defined() ? opt_append_lengths.value().size() : num_total_seqs;
    append_lengths =
        opt_append_lengths.value_or(ShapeTuple(std::vector<int64_t>(/*count=*/nseq, /*value=*/1)));

    CHECK_GE(nseq, num_total_seqs);
    cur_append_lengths = append_lengths;

    for (int64_t i = 0; i < nseq; ++i) {
      ICHECK_LE(i, num_total_seqs);
      if (i == num_total_seqs) {
        // Initialize if it is a new sequence.
        CHECK_GT(append_lengths[i], 0);
        page_table.push_back({});
        seqlen.push_back(0);
        ++num_total_seqs;
      }

      if (append_lengths[i] == 0) {
        continue;
      }

      int64_t cur_npage = (seqlen[i] + page_size - 1) / page_size;
      int64_t tgt_npage = (seqlen[i] + append_lengths[i] + page_size - 1) / page_size;
      for (int64_t page_idx = cur_npage; page_idx < tgt_npage; ++page_idx) {
        AllocatePage(i);
      }
      seqlen[i] += append_lengths[i];
    }

    // Sync auxiliary data structures to device
    SyncDevice(/*sync_current_batch=*/true);
  }

  void Append(PackedFunc f_transpose_append, NDArray k_data, NDArray v_data, int64_t layer_id) {
    // k/v_data: (b, len, nhead, nfeat)
    CHECK_EQ(k_data->ndim, 4);
    CHECK_GT(k_data->shape[1], 0);
    CHECK_EQ(k_data->shape[2], nhead);
    CHECK_EQ(k_data->shape[3], nfeat);

    for (int i = 0; i < 4; ++i) {
      CHECK_EQ(k_data->shape[i], v_data->shape[i]);
    }
    CHECK(k_data.DataType() == pages.DataType());
    CHECK(k_data.DataType() == v_data.DataType());

    int64_t ntoken = 0;
    for (int64_t i = 0; i < num_total_seqs; ++i) {
      ntoken += cur_append_lengths[i];
      if (k_data->shape[0] > 1) {
        CHECK_EQ(cur_append_lengths[i], k_data->shape[1]);
      }
    }
    if (k_data->shape[0] > 1) {
      CHECK_EQ(k_data->shape[0], num_total_seqs);
    }
    CHECK_EQ(ntoken, k_data->shape[0] * k_data->shape[1]);

    // Copy data
    f_transpose_append(pages,  //
                       k_data.CreateView({ntoken, nhead, nfeat}, k_data->dtype),
                       v_data.CreateView({ntoken, nhead, nfeat}, v_data->dtype),
                       page_table_indptr_device.CreateView({num_total_seqs + 1}, dtype_aux),
                       page_table_values_device.CreateView({num_pages_in_use}, dtype_aux),
                       last_page_offset_device.CreateView({num_total_seqs}, dtype_aux),
                       cur_append_length_indptr_device.CreateView({num_total_seqs + 1}, dtype_aux),
                       cur_pos2seqidx_device.CreateView({ntoken}, dtype_aux),  //
                       layer_id);
  }

  void Remove(int64_t seq_id) {
    CHECK_LT(seq_id, num_total_seqs);
    for (int32_t page_id : page_table[seq_id]) {
      FreePage(page_id);
    }
    page_table.erase(page_table.begin() + seq_id);
    seqlen.erase(seqlen.begin() + seq_id);
    num_total_seqs -= 1;
    SyncDevice();
  }

  Array<NDArray> View(PackedFunc f_view) {
    Array<NDArray> kv_values;
    kv_values.reserve(num_total_seqs);

    for (int64_t seq_id = 0; seq_id < num_total_seqs; ++seq_id) {
      NDArray values =
          NDArray::Empty({nlayer, 2, nhead, seqlen[seq_id], nfeat}, pages->dtype, pages->device);
      f_view(pages,                                                                 //
             page_table_indptr_device.CreateView({num_total_seqs + 1}, dtype_aux),  //
             page_table_values_device.CreateView({num_pages_in_use}, dtype_aux),    //
             values, seq_id);
      kv_values.push_back(values);
    }
    return kv_values;
  }

  void PopN(int64_t seq_id, int64_t n) {
    CHECK_LT(seq_id, num_total_seqs);
    CHECK_GE(n, 0);
    CHECK_LE(n, seqlen[seq_id]);

    int64_t cur_npage = (seqlen[seq_id] + page_size - 1) / page_size;
    int64_t tgt_npage = (seqlen[seq_id] - n + page_size - 1) / page_size;
    for (int64_t page_idx = cur_npage - 1; page_idx >= tgt_npage; --page_idx) {
      ICHECK_EQ(page_idx, page_table[seq_id].size() - 1);
      int64_t page_id = page_table[seq_id].back();
      page_table[seq_id].pop_back();
      FreePage(page_id);
    }
    seqlen[seq_id] -= n;
  }

  void Clear() {
    num_total_seqs = 0;
    num_pages_in_use = 0;
    num_pages_allocated = 0;

    free_page_ids.clear();
    page_table.clear();
    seqlen.clear();
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.vm.PagedAttentionKVCache";
  TVM_DECLARE_FINAL_OBJECT_INFO(PagedAttentionKVCacheObj, Object);

 private:
  void AllocatePage(int64_t seq_id) {
    ICHECK_LT(seq_id, num_total_seqs);
    int32_t page_id = GetFreePage();
    page_table[seq_id].push_back(page_id);
    ++num_pages_in_use;
  }

  int32_t GetFreePage() {
    if (!free_page_ids.empty()) {
      int32_t page_id = free_page_ids.back();
      free_page_ids.pop_back();
      return page_id;
    }

    int64_t reserved_num_page_chunks = pages->shape[0];
    if (num_pages_allocated < reserved_num_page_chunks * page_chunk_size) {
      return num_pages_allocated++;
    }
    ICHECK_EQ(num_pages_allocated, reserved_num_page_chunks * page_chunk_size);

    // page chunk grow
    ICHECK_EQ(pages->ndim, 7);
    std::vector<int64_t> new_shape(pages->shape, pages->shape + 7);
    new_shape[0] = reserved_num_page_chunks * 2;
    DLDataType dtype = pages->dtype;
    NDArray new_pages = NDArray::Empty(new_shape, dtype, pages->device);
    new_pages.CreateView(pages.Shape(), dtype).CopyFrom(pages);
    this->pages = new_pages;
    // Also create a larger pos2seqidx
    this->cur_pos2seqidx_device =
        NDArray::Empty({reserved_num_page_chunks * 2 * page_chunk_size * page_size}, dtype_aux,
                       cur_pos2seqidx_device->device);

    return num_pages_allocated++;
  }

  void FreePage(int32_t page_id) {
    free_page_ids.push_back(page_id);
    --num_pages_in_use;
  }

  void DeviceAuxNDArrayGrow() {
    int64_t reserved_nseq = page_table_indptr_device->shape[0] - 1;
    ICHECK_EQ(last_page_offset_device->shape[0], reserved_nseq);
    while (num_total_seqs + 1 > reserved_nseq) {
      reserved_nseq *= 2;
    }

    DLDevice device = page_table_indptr_device->device;
    if (reserved_nseq != page_table_indptr_device->shape[0] - 1) {
      page_table_indptr_device = NDArray::Empty({reserved_nseq + 1}, dtype_aux, device);
      last_page_offset_device = NDArray::Empty({reserved_nseq}, dtype_aux, device);
      cur_append_length_indptr_device = NDArray::Empty({reserved_nseq + 1}, dtype_aux, device);
    }

    if (pages->shape[0] * page_chunk_size > page_table_values_device->shape[0]) {
      page_table_values_device =
          NDArray::Empty({pages->shape[0] * page_chunk_size}, dtype_aux, device);
    }
  }

  void SyncDevice(bool sync_current_batch = false) {
    // Invariant checks
    ICHECK_EQ(page_table.size(), num_total_seqs);
    ICHECK_EQ(seqlen.size(), num_total_seqs);
    for (int64_t seq_id = 0; seq_id < num_total_seqs; ++seq_id) {
      ICHECK(!page_table[seq_id].empty());
    }

    // Grow NDArrays when needed.
    DeviceAuxNDArrayGrow();

    int64_t nbyte_aux = (dtype_aux.bits * dtype_aux.lanes + 7) / 8;
    // Copy page table indptr and values
    std::vector<int32_t> page_table_indptr_host = {0};
    std::vector<int32_t> page_table_values_host;
    for (auto& seq_page_table : page_table) {
      page_table_values_host.insert(page_table_values_host.end(), seq_page_table.begin(),
                                    seq_page_table.end());
      page_table_indptr_host.push_back(page_table_values_host.size());
    }
    std::vector<int32_t> last_page_offset_host;
    last_page_offset_host.reserve(num_total_seqs);
    for (int32_t len : seqlen) {
      ICHECK_GT(len, 0);
      last_page_offset_host.push_back((len - 1) % page_size + 1);
    }

    ICHECK_EQ(page_table_values_host.size(), num_pages_in_use);
    page_table_indptr_device
        .CreateView({static_cast<int64_t>(page_table_indptr_host.size())}, dtype_aux)
        .CopyFromBytes(page_table_indptr_host.data(), page_table_indptr_host.size() * nbyte_aux);
    page_table_values_device
        .CreateView({static_cast<int64_t>(page_table_values_host.size())}, dtype_aux)
        .CopyFromBytes(page_table_values_host.data(), page_table_values_host.size() * nbyte_aux);
    // Copy seqlen
    last_page_offset_device.CreateView({num_total_seqs}, dtype_aux)
        .CopyFromBytes(last_page_offset_host.data(), num_total_seqs * nbyte_aux);

    if (sync_current_batch) {
      std::vector<int32_t> append_length_indptr;
      std::vector<int32_t> pos2seqidx;
      append_length_indptr.reserve(num_total_seqs + 1);
      append_length_indptr.push_back(0);

      for (int64_t i = 0; i < num_total_seqs; ++i) {
        append_length_indptr.push_back(append_length_indptr.back() + cur_append_lengths[i]);
        for (int64_t pos = 0; pos < cur_append_lengths[i]; ++pos) {
          pos2seqidx.push_back(i);
        }
      }
      CHECK_EQ(append_length_indptr.back(), pos2seqidx.size());

      cur_append_length_indptr_device
          .CreateView({static_cast<int64_t>(append_length_indptr.size())}, dtype_aux)
          .CopyFromBytes(append_length_indptr.data(), append_length_indptr.size() * nbyte_aux);
      cur_pos2seqidx_device.CreateView({static_cast<int64_t>(pos2seqidx.size())}, dtype_aux)
          .CopyFromBytes(pos2seqidx.data(), pos2seqidx.size() * nbyte_aux);
    }
  }
};

class PagedAttentionKVCache : public ObjectRef {
 public:
  static PagedAttentionKVCache Create(int64_t reserved_nseq, int64_t total_sequence_length,
                                      int64_t page_size, int64_t nlayer, int64_t nhead,
                                      int64_t nfeat, NDArray init) {
    auto n = make_object<PagedAttentionKVCacheObj>();
    n->num_total_seqs = 0;
    n->num_pages_in_use = 0;
    n->num_pages_allocated = 0;
    n->page_size = page_size;
    n->nlayer = nlayer;
    n->nhead = nhead;
    n->nfeat = nfeat;

    DLDevice device = init->device;
    int64_t reserved_num_page_chunks =
        (((total_sequence_length + page_size - 1) / page_size * nlayer * 2) +
         (n->page_chunk_size - 1)) /
        n->page_chunk_size;
    n->pages = NDArray::Empty(
        {reserved_num_page_chunks, nlayer, n->page_chunk_size, 2, nhead, page_size, nfeat},
        init->dtype, device);

    n->page_table_indptr_device = NDArray::Empty({reserved_nseq + 1}, n->dtype_aux, device);
    n->page_table_values_device =
        NDArray::Empty({reserved_num_page_chunks * n->page_chunk_size}, n->dtype_aux, device);
    n->last_page_offset_device = NDArray::Empty({reserved_nseq}, n->dtype_aux, device);
    n->cur_append_length_indptr_device = NDArray::Empty({reserved_nseq + 1}, n->dtype_aux, device);
    n->cur_pos2seqidx_device = NDArray::Empty(
        {reserved_num_page_chunks * n->page_chunk_size * page_size}, n->dtype_aux, device);

    return PagedAttentionKVCache(n);
  }

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PagedAttentionKVCache, ObjectRef, PagedAttentionKVCacheObj);
};

TVM_REGISTER_OBJECT_TYPE(PagedAttentionKVCacheObj);

//-------------------------------------------------
//  Register runtime functions
//-------------------------------------------------

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_create")
    .set_body_typed([](ShapeTuple cache_config, int64_t nlayer, int64_t nhead, int64_t nfeat,
                       NDArray init) {
      CHECK_EQ(cache_config.size(), 3);
      return PagedAttentionKVCache::Create(cache_config[0], cache_config[1], cache_config[2],
                                           nlayer, nhead, nfeat, init);
    });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_prepare")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.size() == 1 || args.size() == 2);
      PagedAttentionKVCache cache = args[0];
      Optional<ShapeTuple> append_lengths;
      if (args.size() == 2) {
        append_lengths = ShapeTuple(args[1]);
      }
      cache->Prepare(append_lengths);
      *rv = cache;
    });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_append")
    .set_body_typed([](PagedAttentionKVCache cache, PackedFunc f_transpose_append, NDArray k_data,
                       NDArray v_data, int64_t layer_id) {
      cache->Append(f_transpose_append, k_data, v_data, layer_id);
      return cache;
    });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_remove")
    .set_body_typed([](PagedAttentionKVCache cache, int64_t seq_id) { cache->Remove(seq_id); });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_view_testing")
    .set_body_typed([](PagedAttentionKVCache cache, PackedFunc f_view) {
      return cache->View(f_view);
    });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_popn")
    .set_body_typed([](PagedAttentionKVCache cache, int64_t seq_id, int64_t n) {
      return cache->PopN(seq_id, n);
    });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_clear")
    .set_body_typed([](PagedAttentionKVCache cache) { return cache->Clear(); });

TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_attention")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.size() == 5 || args.size() == 8);
      bool apply_rotary = false;
      double rotary_scale = 1.0;
      double rotary_theta = 1e4;
      if (args.size() == 8) {
        apply_rotary = args[4];
        rotary_scale = args[5];
        rotary_theta = args[6];
      }
      PagedAttentionKVCache cache = args[0];
      cache->Attention(/*f_attention=*/args[1], /*q_data=*/args[2], /*layer_id=*/args[3],
                       /*output=*/args[args.size() - 1], apply_rotary, rotary_scale, rotary_theta);
    });

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
