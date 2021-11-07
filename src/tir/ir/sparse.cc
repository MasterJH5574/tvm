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
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/sparse.h>

namespace tvm {
namespace tir {

// Axis
TVM_REGISTER_GLOBAL("tir.sparse.GetAxisName").set_body_typed([](Axis axis) {
  return axis->GetName();
});

TVM_REGISTER_GLOBAL("tir.sparse.GetAxisLength").set_body_typed([](Axis axis) {
  return axis->GetLength();
});

TVM_REGISTER_GLOBAL("tir.sparse.GetAxisIndexType").set_body_typed([](Axis axis) {
  return DLDataType2String(axis->GetIndexType());
});

// DenseFixedAxis
DenseFixedAxis::DenseFixedAxis(String name, PrimExpr length) {
  ObjectPtr<DenseFixedAxisNode> node = make_object<DenseFixedAxisNode>();
  node->name = std::move(name);
  node->length = std::move(length);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(DenseFixedAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.DenseFixedAxis").set_body_typed([](String name, PrimExpr length) {
  return DenseFixedAxis(name, length);
});

// DenseVariableAxis
DenseVariableAxis::DenseVariableAxis(String name, PrimExpr length, Buffer indptr) {
  ObjectPtr<DenseVariableAxisNode> node = make_object<DenseVariableAxisNode>();
  node->name = std::move(name);
  node->length = std::move(length);
  node->indptr = std::move(indptr);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(DenseVariableAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.DenseVariableAxis")
    .set_body_typed([](String name, PrimExpr length, Buffer indptr) {
      return DenseVariableAxis(name, length, indptr);
    });

// SparseFixedAxis
SparseFixedAxis::SparseFixedAxis(String name, PrimExpr length, Buffer indices, PrimExpr num_cols) {
  ObjectPtr<SparseFixedAxisNode> node = make_object<SparseFixedAxisNode>();
  node->name = std::move(name);
  node->length = std::move(length);
  node->indices = std::move(indices);
  node->num_cols = std::move(num_cols);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(SparseFixedAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.SparseFixedAxis")
    .set_body_typed([](String name, PrimExpr length, Buffer indices, PrimExpr num_cols) {
      return SparseFixedAxis(name, length, indices, num_cols);
    });

// SparseVariableAxis
SparseVariableAxis::SparseVariableAxis(String name, PrimExpr length, Buffer indptr,
                                       Buffer indices) {
  ObjectPtr<SparseVariableAxisNode> node = make_object<SparseVariableAxisNode>();
  node->name = std::move(name);
  node->length = std::move(length);
  node->indptr = std::move(indptr);
  node->indices = std::move(indices);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(SparseVariableAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.SparseVariableAxis")
    .set_body_typed([](String name, PrimExpr length, Buffer indptr, Buffer indices) {
      return SparseVariableAxis(name, length, indptr, indices);
    });

// AxisTree
AxisTree::AxisTree(Array<String> axis_names, Array<Optional<String>> axis_parent_names) {
  CHECK_EQ(axis_names.size(), axis_parent_names.size())
      << "ValueError: The axis_names array should have the same length as "
         "axis_parent_names "
         "array.";
  ObjectPtr<AxisTreeNode> node = make_object<AxisTreeNode>();
  for (size_t i = 0; i < axis_names.size(); i++) {
    // update parent map & children map
    String axis_name = axis_names[i];
    Optional<String> parent_name = axis_parent_names[i];
    node->parent[axis_name] = parent_name;
    if (node->children.find(parent_name) != node->children.end()) {
      node->children[parent_name].push_back(axis_name);
    } else {
      Array<String> children;
      children.push_back(axis_name);
      node->children[parent_name] = std::move(children);
    }
  }
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(AxisTreeNode);

TVM_REGISTER_GLOBAL("tir.sparse.AxisTree")
    .set_body_typed([](Array<String> axis_names, Array<Optional<String>> axis_parent_names) {
      return AxisTree(axis_names, axis_parent_names);
    });

// SparseBuffer
SparseBuffer::SparseBuffer(Array<Axis> axes, Buffer data, String name) {
  ObjectPtr<SparseBufferNode> node = make_object<SparseBufferNode>();
  CHECK_GT(static_cast<int>(axes.size()), 0)
      << "ValueError: A SparseBuffer should have at least one dimension";
  node->axes = std::move(axes);
  node->data = std::move(data);
  node->name = std::move(name);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(SparseBufferNode);

TVM_REGISTER_GLOBAL("tir.sparse.SparseBuffer")
    .set_body_typed([](Array<Axis> axes, Buffer data, String name) {
      return SparseBuffer(axes, data, name);
    });

// SpIterVar
SpIterVar::SpIterVar(Var var, PrimExpr max_extent, SpIterKind kind, bool is_reduction,
                     Optional<Axis> axis) {
  ObjectPtr<SpIterVarNode> node = make_object<SpIterVarNode>();

  arith::Analyzer ana;
  if (axis.defined()) {
    CHECK(ana.CanProveEqual(axis.value()->length, max_extent));
  }
  if (kind != SpIterKind::kDenseFixed) {
    CHECK(axis.defined()) << "ValueError: To create a SpIterVar that is not fixed-dense, one must "
                             "specify the axis over which the SpIterVar iterates";
    const char* err_str = "ValueError: The given kind doesn't match the type of the given axis";
    if (kind == SpIterKind::kDenseVariable) {
      CHECK(axis.value()->IsInstance<DenseFixedAxisNode>()) << err_str;
    } else if (kind == SpIterKind::kSparseFixed) {
      CHECK(axis.value()->IsInstance<SparseFixedAxisNode>()) << err_str;
    } else if (kind == SpIterKind::kSparseVariable) {
      CHECK(axis.value()->IsInstance<SparseVariableAxisNode>()) << err_str;
    }
  }

  node->var = Var(std::move(var));
  node->max_extent = std::move(max_extent);
  node->kind = kind;
  node->is_reduction = is_reduction;
  node->axis = std::move(axis);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(SpIterVarNode);

TVM_REGISTER_GLOBAL("tir.sparse.SpIterVar")
    .set_body_typed([](Var var, PrimExpr max_extent, int kind, bool is_reduction,
                       Optional<Axis> axis) {
      return SpIterVar(var, max_extent, SpIterKind(kind), is_reduction, axis);
    });

}  // namespace tir
}  // namespace tvm
