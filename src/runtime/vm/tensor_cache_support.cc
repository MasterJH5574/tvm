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
 * \file src/runtime/vm/tensor_cache_support.cc
 * \brief Runtime to support tensor cache file loading.
 *
 * This file provides a minimum support for tensor cache file loading.
 *
 * The main focus of this implementation is to enable loading
 * with minimum set of intermediate files while also being
 * compatible to some of the multi-shard files that are more
 * friendly in some of the environments.
 *
 * Tensor cache also provides a way to do system-wide
 * parameter sharing across multiple VMs.
 *
 * There are likely other ways to load the parameters ndarray-ache.
 * We will keep the impact minimum by puting it as a private
 * runtime builtin provide as in this file.
 */
#define PICOJSON_USE_INT64
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <picojson.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/tensor.h>
#include <tvm/runtime/vm/tensor_cache_support.h>

#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <vector>

#include "../../support/utils.h"
#include "../file_utils.h"

namespace tvm {
namespace runtime {
namespace vm {

template <typename ExpectedType>
inline ExpectedType AsType(const picojson::value& json) {
  ICHECK(json.is<ExpectedType>());
  return json.get<ExpectedType>();
}

template <typename ValueType>
inline ValueType GetValue(const picojson::object& json, const std::string& key) {
  return AsType<ValueType>(json.at(key));
}

TensorCacheMetadata::FileRecord::ParamRecord JSONAsParamRecord(const picojson::object& json) {
  std::vector<ffi::Shape::index_type> shape;
  {
    picojson::array shape_json = GetValue<picojson::array>(json, "shape");
    shape.reserve(shape_json.size());
    for (const picojson::value& d : shape_json) {
      shape.push_back(AsType<int64_t>(d));
    }
  }
  TensorCacheMetadata::FileRecord::ParamRecord result;
  std::string dtype = GetValue<std::string>(json, "dtype");
  result.name = GetValue<std::string>(json, "name");
  result.dtype = DataType(ffi::StringToDLDataType(dtype));
  result.format = GetValue<std::string>(json, "format");
  result.nbytes = GetValue<int64_t>(json, "nbytes");
  result.byte_offset = GetValue<int64_t>(json, "byteOffset");
  result.shape = ffi::Shape(std::move(shape));
  return result;
}

TensorCacheMetadata::FileRecord JSONAsFileRecord(const picojson::object& json) {
  picojson::array records = GetValue<picojson::array>(json, "records");
  TensorCacheMetadata::FileRecord result;
  result.data_path = GetValue<std::string>(json, "dataPath");
  result.format = GetValue<std::string>(json, "format");
  result.nbytes = GetValue<int64_t>(json, "nbytes");
  result.records.reserve(records.size());
  for (const picojson::value& item : records) {
    result.records.push_back(JSONAsParamRecord(AsType<picojson::object>(item)));
  }
  return result;
}

TensorCacheMetadata JSONAsTensorCacheMetadata(const picojson::object& json) {
  picojson::array records = GetValue<picojson::array>(json, "records");
  TensorCacheMetadata result;
  result.records.reserve(records.size());
  for (const picojson::value& item : records) {
    result.records.push_back(JSONAsFileRecord(AsType<picojson::object>(item)));
  }
  return result;
}

TensorCacheMetadata TensorCacheMetadata::LoadFromStr(const std::string& json_str,
                                                     const std::string& path) {
  picojson::value json_info;
  {
    std::string err = picojson::parse(json_info, json_str);
    if (!err.empty()) {
      LOG(FATAL) << "Failed to parse JSON: err. The JSON string is:" << json_str;
    }
    CHECK(json_info.is<picojson::object>())
        << "ValueError: The given string is not a JSON object: " << json_str;
  }
  TensorCacheMetadata result = JSONAsTensorCacheMetadata(AsType<picojson::object>(json_info));
  result.path = path;
  return result;
}

TVM_DLL TensorCacheMetadata TensorCacheMetadata::Load(const std::string& path) {
  picojson::value json_info;
  {
    std::string json_str;
    LoadBinaryFromFile(path + "/tensor-cache.json", &json_str);
    std::string err = picojson::parse(json_info, json_str);
    if (!err.empty()) {
      LOG(FATAL) << "Failed to parse JSON: err. The JSON string is:" << json_str;
    }
    CHECK(json_info.is<picojson::object>())
        << "ValueError: The given string is not a JSON object: " << json_str;
  }
  TensorCacheMetadata result = JSONAsTensorCacheMetadata(AsType<picojson::object>(json_info));
  result.path = path;
  return result;
}

void CopyTensorFromBytes(Tensor param, const void* data, size_t nbytes,
                         ffi::Optional<Tensor>* staging_buffer) {
  Device device = param->device;
  if (device.device_type != kDLOpenCL || staging_buffer == nullptr) {
    param.CopyFromBytes(data, nbytes);
    return;
  }
  // Special handle for OpenCL runtime.
  // It creates a host side memory mirror, for every cl_mem that tries to copy data from host
  // which can cause memory issue. Her we use a large staging buffer to postpone deallocation
  if (staging_buffer->defined()) {
    size_t curr_size = runtime::GetDataSize(*(staging_buffer->value().operator->()));
    if (curr_size < nbytes) {
      *staging_buffer = std::nullopt;
    }
  }
  if (!staging_buffer->defined()) {
    *staging_buffer = Tensor::Empty(param.Shape(), param->dtype, param->device);
  }
  Tensor staging_view = staging_buffer->value().CreateView(param.Shape(), param->dtype);
  staging_view.CopyFromBytes(data, nbytes);
  param.CopyFrom(staging_view);
  DeviceAPI::Get(device)->StreamSync(device, nullptr);
}

Tensor TensorCacheMetadata::FileRecord::ParamRecord::Load(
    Device device, const std::string* raw_data, ffi::Optional<Tensor>* staging_buffer) const {
  Tensor arr = Tensor::Empty(shape, dtype, device);
  if (dtype == DataType::Float(32) && format == "f32-to-bf16") {
    // decode bf16 to f32
    std::vector<uint16_t> buffer(nbytes / 2);
    std::vector<uint32_t> decoded(nbytes / 2);
    std::memcpy(buffer.data(), raw_data->data() + byte_offset, nbytes);
    for (size_t i = 0; i < buffer.size(); ++i) {
      decoded[i] = static_cast<uint32_t>(buffer[i]) << 16;
    }
    CopyTensorFromBytes(arr, decoded.data(), decoded.size() * sizeof(uint32_t), staging_buffer);
  } else {
    CopyTensorFromBytes(arr, raw_data->data() + byte_offset, nbytes, staging_buffer);
  }
  return arr;
}

// class ViewHelper {
//   public:
//    explicit ViewHelper(Tensor source) : source_(source) {}
//    void AllocData(DLTensor* tensor, int64_t extra_byte_offset, int64_t nbytes) {
//      if (extra_byte_offset == 0) {
//       tensor->data = source_->data;
//      LOG(INFO) << "ViewHelper returning early because extra_byte_offset is 0";
//       return;
//      }
//      static ffi::Function metal_view_with_offset = ffi::Function::GetGlobalRequired("metal.MTLBufferViewWithOffset");
//     LOG(INFO) << "calling metal.MTLBufferViewWithOffset";
//     //  tensor->data = source_->data;
//      tensor->data = metal_view_with_offset(source_->data, extra_byte_offset, nbytes, source_->device).cast<void*>();
//     LOG(INFO) << "calling metal.MTLBufferViewWithOffset returned";
//    }

//    void FreeData(DLTensor* tensor) {}

//   private:
//    Tensor source_;
//  };

Tensor LoadBinaryFromFileToMemory(const std::string& file_name, Device device) {
  std::ifstream fs(file_name, std::ios::in | std::ios::binary);
  ICHECK(!fs.fail()) << "Cannot open " << file_name;
  // get its size:
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  // LOG(INFO) << "LoadBinaryFromFileToMemory: " << file_name << ", size: " << size;
  // Tensor arr = Tensor::Empty({static_cast<int64_t>(size)}, DataType::UInt(8), Device{DLDeviceType::kDLCPU, 0});
  Tensor arr = Tensor::Empty({static_cast<int64_t>(size)}, DataType::UInt(8), device);
  void* data_ptr = arr->data;
  static ffi::Function get_mtl_buffer_raw_data_pointer = ffi::Function::GetGlobalRequired("metal.GetMTLBufferRawDataPointer");
  void* raw_data_ptr = get_mtl_buffer_raw_data_pointer(data_ptr).cast<void*>();
  // LOG(INFO) << "LoadBinaryFromFileToMemory before read, raw data pointer: " << raw_data_ptr;
  fs.read(static_cast<char*>(raw_data_ptr), size);
  // LOG(INFO) << "LoadBinaryFromFileToMemory after read";
  return arr;
}

Tensor ParamRecordLoadFromTensor(const TensorCacheMetadata::FileRecord::ParamRecord& param_record, Tensor raw_data) {
  // LOG(INFO) << "Creating view for " << param_record.name << " with shape " << param_record.shape << ", dtype " << param_record.dtype << ", nbytes " << param_record.nbytes << ", byte_offset " << param_record.byte_offset;
  Tensor view = raw_data.CreateView(param_record.shape, param_record.dtype, param_record.byte_offset);
  // Tensor new_tensor = Tensor::Empty(param_record.shape, param_record.dtype, raw_data->device);
  // new_tensor.CopyFrom(view);
  // return new_tensor;
  // ffi::Function dump_tensor = ffi::Function::GetGlobalRequired("dump_tensor");
  // dump_tensor(view, param_record.name);
  // Tensor view = Tensor::FromNDAlloc(ViewHelper(raw_data), param_record.shape,
  //                           param_record.dtype, raw_data->device, param_record.byte_offset, param_record.nbytes);
  // LOG(INFO) << "Created view";
  return view;
}

// struct TensorCacheThreadEntry {
//    std::vector<Tensor> raw_tensors;

//    static TensorCacheThreadEntry* ThreadLocal() {
//     thread_local static TensorCacheThreadEntry entry;
//     return &entry;
//    }
//  };

TVM_DLL ffi::Array<Tensor> TensorCacheMetadata::FileRecord::Load(
    Device device,
    const std::string& path_prefix,  //
    std::string* raw_data_buffer,    //
    ffi::Optional<Tensor>* staging_buffer) const {
  // LoadBinaryFromFile(path_prefix + "/" + this->data_path, raw_data_buffer);
  Tensor data = LoadBinaryFromFileToMemory(path_prefix + "/" + this->data_path, device);
  // TensorCacheThreadEntry::ThreadLocal()->raw_tensors.push_back(data);
  // LOG(INFO) << "file loaded";
  // std::this_thread::sleep_for(std::chrono::seconds(10));
  CHECK_EQ(this->format, "raw-shard") << "ValueError: Only `raw-shard` format is supported";
  // CHECK_EQ(this->nbytes, raw_data_buffer->length())
  //     << "ValueError: Encountered an corrupted parameter shard. It means it is not downloaded "
  //        "completely or downloading is interrupted. Please try to download again.";
  ffi::Array<Tensor> result;
  result.reserve(this->records.size());
  // LOG(INFO) << "records size: " << this->records.size();
  for (const ParamRecord& nd_rec : this->records) {
    // result.push_back(nd_rec.Load(device, raw_data_buffer, staging_buffer));
    // LOG(INFO) << "Loading param: " << nd_rec.name;
    result.push_back(ParamRecordLoadFromTensor(nd_rec, data));
    // LOG(INFO) << "Loaded param: " << nd_rec.name;

    // ffi::Function dump_tensor = ffi::Function::GetGlobalRequired("dump_tensor");
    // dump_tensor(result.back(), nd_rec.name);
  }
  // Todo: TE legalize add offset_factor
  // LOG(INFO) << "FileRecord::Load returned";
  return result;
}

/*!
 * A Tensor cache to store pre-loaded arrays in the system.
 */
class TensorCache {
 public:
  static TensorCache* Global() {
    static TensorCache* inst = new TensorCache();
    return inst;
  }

  static void Update(ffi::String name, Tensor arr, bool override) {
    TensorCache* pool = Global();
    if (!override) {
      ICHECK_EQ(pool->pool_.count(name), 0) << "Name " << name << " already exists in the cache";
    }
    pool->pool_.Set(name, arr);
  }

  static ffi::Optional<Tensor> Get(ffi::String name) {
    TensorCache* pool = Global();
    auto it = pool->pool_.find(name);
    if (it != pool->pool_.end()) {
      return (*it).second;
    } else {
      return std::nullopt;
    }
  }

  static void Remove(ffi::String name) {
    TensorCache* pool = Global();
    pool->pool_.erase(name);
  }

  static void Clear() { Global()->pool_.clear(); }

  /*!
   * \brief Load parameters from path and append them.
   * \param cache_path The cache to path.
   * \param device_type The type of device to be loaded.
   * \param device_id The device id.
   */
  static void Load(const std::string& cache_path, int device_type, int device_id) {
    DLDevice device{static_cast<DLDeviceType>(device_type), device_id};
    TensorCacheMetadata metadata = TensorCacheMetadata::Load(cache_path);
    ffi::Optional<Tensor> staging_buffer;
    std::string raw_data;
    ffi::Array<Tensor> params;
    for (const TensorCacheMetadata::FileRecord& shard_rec : metadata.records) {
      try {
        params = shard_rec.Load(device, cache_path, &raw_data, &staging_buffer);
      } catch (const dmlc::Error& e) {
        LOG(FATAL) << "ValueError: Error when loading parameters from " << shard_rec.data_path
                   << ": " << e.what();
      }
      int num_params = params.size();
      for (int i = 0; i < num_params; ++i) {
        Update(shard_rec.records[i].name, params[i], true);
      }
    }
  }

  static void LoadByNames(const std::string& cache_path, const ffi::Array<ffi::String>& names,
                          int device_type, int device_id) {
    DLDevice device{static_cast<DLDeviceType>(device_type), device_id};
    TensorCacheMetadata metadata = TensorCacheMetadata::Load(cache_path);
    ffi::Optional<Tensor> staging_buffer;
    std::string raw_data;
    ffi::Array<Tensor> params;

    auto f_shard_rec_contains_name = [](const TensorCacheMetadata::FileRecord& shard_rec,
                                        const ffi::String& name) {
      for (const TensorCacheMetadata::FileRecord::ParamRecord& param_rec : shard_rec.records) {
        if (param_rec.name == name) {
          return true;
        }
      }
      return false;
    };

    for (const TensorCacheMetadata::FileRecord& shard_rec : metadata.records) {
      // if shard_rec doesn't contain any of the names, skip
      bool contains_any_name = false;
      for (const ffi::String& name : names) {
        if (f_shard_rec_contains_name(shard_rec, name)) {
          contains_any_name = true;
          break;
        }
      }
      if (!contains_any_name) {
        continue;
      }

      // LOG(INFO) << "processing shard_rec: " << shard_rec.data_path;
      try {
        params = shard_rec.Load(device, cache_path, &raw_data, &staging_buffer);
      } catch (const dmlc::Error& e) {
        LOG(FATAL) << "ValueError: Error when loading parameters from " << shard_rec.data_path
                   << ": " << e.what();
      }
      int num_params = params.size();
      for (int i = 0; i < num_params; ++i) {
        if (std::find(names.begin(), names.end(), ffi::String(shard_rec.records[i].name)) ==
            names.end()) {
          // LOG(INFO) << "[skip] loaded param: " << shard_rec.records[i].name
          //           << " shape=" << shard_rec.records[i].shape
          //           << ", dtype=" << shard_rec.records[i].dtype
          //           << ", size=" << shard_rec.records[i].nbytes << " bytes";
          continue;
        }
        Update(shard_rec.records[i].name, params[i], true);
        // LOG(INFO) << "[keep] loaded param: " << shard_rec.records[i].name
        //           << " shape=" << shard_rec.records[i].shape
        //           << ", dtype=" << shard_rec.records[i].dtype
        //           << ", size=" << shard_rec.records[i].nbytes << " bytes";
      }
      // LOG(INFO) << "before clear raw_data";
      // std::this_thread::sleep_for(std::chrono::seconds(10));
      raw_data.clear();
      raw_data.shrink_to_fit();
      // LOG(INFO) << "after clear raw_data";
      // std::this_thread::sleep_for(std::chrono::seconds(10));
    }
  }

 private:
  ffi::Map<ffi::String, Tensor> pool_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("vm.builtin.tensor_cache.get", TensorCache::Get)
      .def_packed("vm.builtin.tensor_cache.update",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    CHECK(args.size() == 2 || args.size() == 3);
                    ffi::String name = args[0].cast<ffi::String>();
                    bool is_override = args.size() == 2 ? false : args[2].cast<bool>();

                    Tensor arr;
                    if (auto opt_nd = args[1].as<Tensor>()) {
                      arr = opt_nd.value();
                    } else {
                      // We support converting DLTensors to Tensors as RPC references are always
                      // DLTensors
                      auto tensor = args[1].cast<DLTensor*>();
                      std::vector<int64_t> shape;
                      for (int64_t i = 0; i < tensor->ndim; i++) {
                        shape.push_back(tensor->shape[i]);
                      }
                      arr = Tensor::Empty(shape, tensor->dtype, tensor->device);
                      arr.CopyFrom(tensor);
                      DeviceAPI::Get(arr->device)->StreamSync(arr->device, nullptr);
                    }

                    TensorCache::Update(name, arr, is_override);
                  })
      .def("vm.builtin.tensor_cache.remove", TensorCache::Remove)
      .def("vm.builtin.tensor_cache.clear", TensorCache::Clear)
      .def("vm.builtin.tensor_cache.load", TensorCache::Load)
      .def("vm.builtin.tensor_cache.load_by_names", TensorCache::LoadByNames);
}

// This param module node can be useful to get param dict in RPC mode
// when the remote already have loaded parameters from file.
class ParamModuleNode : public ffi::ModuleObj {
 public:
  const char* kind() const final { return "param_module"; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    if (name == "get_params") {
      auto params = params_;
      return ffi::Function([params](ffi::PackedArgs args, ffi::Any* rv) { *rv = params; });
    } else {
      return ffi::Function();
    }
  }

  static ffi::Array<Tensor> GetParams(const ffi::String& prefix, int num_params) {
    ffi::Array<Tensor> params;
    for (int i = 0; i < num_params || num_params == -1; ++i) {
      std::string name = prefix + "_" + std::to_string(i);
      auto opt = TensorCache::Get(name);
      if (opt) {
        params.push_back(opt.value());
      } else {
        if (num_params == -1) return params;
        LOG(FATAL) << "Cannot find " << name << " in cache";
      }
    }
    return params;
  }

  static ffi::Array<Tensor> GetParamByName(const ffi::Array<ffi::String>& names) {
    ffi::Array<Tensor> result;
    result.reserve(names.size());
    for (const ffi::String& name : names) {
      if (ffi::Optional<Tensor> opt = TensorCache::Get(name)) {
        result.push_back(opt.value());
      } else {
        LOG(FATAL) << "ValueError: Cannot find parameter in cache: " << name;
      }
    }
    return result;
  }

  static ffi::Module Create(const std::string& prefix, int num_params) {
    auto n = ffi::make_object<ParamModuleNode>();
    n->params_ = GetParams(prefix, num_params);
    return ffi::Module(n);
  }

  static ffi::Module CreateByName(const ffi::Array<ffi::String>& names) {
    auto n = ffi::make_object<ParamModuleNode>();
    n->params_ = GetParamByName(names);
    return ffi::Module(n);
  }

 private:
  ffi::Array<Tensor> params_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("vm.builtin.param_module_from_cache", ParamModuleNode::Create)
      .def("vm.builtin.param_module_from_cache_by_name", ParamModuleNode::CreateByName)
      .def("vm.builtin.param_array_from_cache", ParamModuleNode::GetParams)
      .def("vm.builtin.param_array_from_cache_by_name", ParamModuleNode::GetParamByName)
      .def_packed("vm.builtin.param_array_from_cache_by_name_unpacked",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    ffi::Array<ffi::String> names;
                    names.reserve(args.size());
                    for (int i = 0; i < args.size(); ++i) {
                      if (!args[i].try_cast<ffi::String>()) {
                        LOG(FATAL) << "ValueError: Expect string as input, but get "
                                   << args[i].GetTypeKey() << " at " << i;
                      }
                      names.push_back(args[i].cast<ffi::String>());
                    }
                    *rv = ParamModuleNode::GetParamByName(names);
                  });
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
