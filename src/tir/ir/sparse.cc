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
 * \file sparse.cc
 * \brief buffers and formats in sparse tir.
 */
#include <tvm/tir/buffer.h>
#include <tvm/tir/sparse.h>

namespace tvm {
namespace tir {

namespace sparse {

DenseFixedAxis::DenseFixedAxis(String name, PrimExpr length) {
  ObjectPtr<DenseFixedAxisNode> node = make_object<DenseFixedAxisNode>();
  node->name = std::move(name);
  node->length = std::move(length);
  data_ = std::move(node);
}

DenseVariableAxis::DenseVariableAxis(String name, PrimExpr length, Buffer indptr) {
  ObjectPtr<DenseVariableAxisNode> node = make_object<DenseVariableAxisNode>();
  node->name = std::move(name);
  node->length = std::move(length);
  node->indptr = std::move(indptr);
  data_ = std::move(node);
}

SparseFixedAxis::SparseFixedAxis(String name, PrimExpr length, Buffer indices, PrimExpr num_cols) {
  ObjectPtr<SparseFixedAxisNode> node = make_object<SparseFixedAxisNode>();
  node->name = std::move(name);
  node->length = std::move(length);
  node->indices = std::move(indices);
  node->num_cols = std::move(num_cols);
  data_ = std::move(node);
}

SparseVariableAxis::SparseVariableAxis(String name, PrimExpr length, Buffer indptr,
                                       Buffer indices) {
  ObjectPtr<SparseVariableAxisNode> node = make_object<SparseVariableAxisNode>();
  node->name = std::move(name);
  node->length = std::move(length);
  node->indptr = std::move(indptr);
  node->indices = std::move(indices);
  data_ = std::move(node);
}

SparseBuffer::SparseBuffer(AxisTree root, Array<Axis> axes, int ndim, Buffer data) {
  ObjectPtr<SparseBufferNode> node = make_object<SparseBufferNode>();
  node->root = std::move(root);
  node->axes = std::move(axes);
  node->ndim = ndim;
  node->data = std::move(data);
  data_ = std::move(node);
}

// TODO(zihao/ruihang)
}  // namespace sparse

}  // namespace tir
}  // namespace tvm
