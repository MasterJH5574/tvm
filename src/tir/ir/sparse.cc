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

/******** Attributes of sparse axis. ********/

TVM_REGISTER_GLOBAL("tir.sparse.GetAxisName").set_body_typed([](Axis axis) {
  return axis->GetName();
});

TVM_REGISTER_GLOBAL("tir.sparse.GetAxisLength").set_body_typed([](Axis axis) {
  return axis->GetLength();
});

TVM_REGISTER_GLOBAL("tir.sparse.GetAxisIndexType").set_body_typed([](Axis axis) {
  return DLDataType2String(axis->GetIndexType());
});

/******** DenseFixedAxis ********/

/*! \brief Default constructor of DenseFixedAxis */
DenseFixedAxis::DenseFixedAxis(String name, PrimExpr length) {
  ObjectPtr<DenseFixedAxisNode> node = make_object<DenseFixedAxisNode>();
  node->name = std::move(name);
  node->length = std::move(length);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(DenseFixedAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.DenseFixedAxis").set_body_typed([](String name, PrimExpr length) {
  return DenseFixedAxis(std::move(name), std::move(length));
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DenseFixedAxisNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const DenseFixedAxisNode*>(node.get());
      p->stream << "dense_fixed(" << op->name << ", " << op->length << ")";
    });

/******** DenseVariableAxis ********/

/*! \brief Default constuctor of DenseVariableAxis */
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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DenseVariableAxisNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const DenseVariableAxisNode*>(node.get());
      p->stream << "dense_variable(" << op->name << ", " << op->length << ", " << op->indptr->name
                << ")";
    });

/******** DenseFromSparseAxis ********/

/*! \brief Default constructor of DenseFromSparseAxis */
DenseFromSparseAxis::DenseFromSparseAxis(SparseAxis base) {
  ObjectPtr<DenseFromSparseAxisNode> node = make_object<DenseFromSparseAxisNode>();
  node->name = base->name + "_dense";
  node->length = base->length;
  node->is_derived_axis = true;
  node->base = std::move(base);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(DenseFromSparseAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.DenseFromSparseAxis").set_body_typed([](SparseAxis base) {
  return DenseFromSparseAxis(std::move(base));
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DenseFromSparseAxisNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const DenseFromSparseAxisNode*>(node.get());
      p->stream << "dense_from_sparse(" << op->base->name << ")";
    });

/******** FusedAxis ********/

/*! \brief Default constructor of FusedAxis */
FusedAxis::FusedAxis(Array<Axis> group, int index) {
  CHECK(index < int(group.size())) << "Index " << index << "exceeds the size of fused axes group.";

  // TODO(zihao): check whether it valid to fuse axes in the group.

  ObjectPtr<FusedAxisNode> node = make_object<FusedAxisNode>();
  std::string fused_name = group[0]->name;
  for (int i = 1; i < group.size(); ++i) {
    fused_name += group[i]->name;
  }
  node->name = "fused_" + fused_name + "_" + group[index]->name;

  if (const auto* df_axis = group[index].as<DenseFixedAxisNode>()) {
    node->length = df_axis->length;
  } else if (const auto* sf_axis = group[index].as<SparseFixedAxisNode>()) {
    // TODO(zihao): accumulate previous dimensions.
  } else if (const auto* dv_axis = group[index].as<DenseVariableAxisNode>()) {
    node->length = dv_axis->nnz();
  } else if (const auto* sv_axis = group[index].as<SparseVariableAxisNode>()) {
    node->length = sv_axis->nnz();
  }

  node->is_derived_axis = true;
  node->group = std::move(group);
  node->index = index;
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(FusedAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.FusedAxis").set_body_typed([](Array<Axis> group, int index) {
  return FusedAxis(std::move(group), index);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FusedAxisNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const FusedAxisNode*>(node.get());
      p->stream << "fused(";
      bool first = true;
      for (auto&& orig_axis : op->group) {
        if (first) {
          first = false;
        } else {
          p->stream << ", ";
        }
        p->stream << orig_axis->name;
      }
      p->stream << ")";
    });

/******** SparseFixedAxis ********/

/*! \brief Default constructor of SparseFixedAxis */
SparseFixedAxis::SparseFixedAxis(String name, PrimExpr length, Buffer indices, PrimExpr nnz_cols) {
  ObjectPtr<SparseFixedAxisNode> node = make_object<SparseFixedAxisNode>();
  node->name = std::move(name);
  node->length = std::move(length);
  node->indices = std::move(indices);
  node->nnz_cols = std::move(nnz_cols);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(SparseFixedAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.SparseFixedAxis")
    .set_body_typed([](String name, PrimExpr length, Buffer indices, PrimExpr nnz_cols) {
      return SparseFixedAxis(name, length, indices, nnz_cols);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SparseFixedAxisNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SparseFixedAxisNode*>(node.get());
      p->stream << "sparse_fixed(" << op->name << ", " << op->length << ", " << op->nnz_cols << ", "
                << op->indices->name << ")";
    });

/******** SparseVariableAxis ********/

/*! \brief Default constructor of SparseVariableAxis */
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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SparseVariableAxisNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SparseVariableAxisNode*>(node.get());
      p->stream << "sparse_variable(" << op->name << ", " << op->length << ", " << op->indptr->name
                << ", " << op->indices->name << ")";
    });

/******** AxisTree ********/

/*! \brief Default constructor of AxisTree */
AxisTree::AxisTree(Array<String> axis_names, Array<Optional<String>> axis_parent_names) {
  CHECK_EQ(axis_names.size(), axis_parent_names.size())
      << "ValueError: The axis_names array should have the same length as "
         "axis_parent_names "
         "array.";
  ObjectPtr<AxisTreeNode> node = make_object<AxisTreeNode>();
  Map<String, String> parent;
  Map<String, Array<String>> children;
  for (size_t i = 0; i < axis_names.size(); i++) {
    // update parent map & children map
    String axis_name = axis_names[i];
    String parent_name("root");
    if (axis_parent_names[i].defined()) {
      parent_name = axis_parent_names[i].value();
    }
    parent.Set(axis_name, parent_name);

    auto it = children.find(parent_name);
    if (it != children.end()) {
      Array<String> value = (*it).second;
      value.push_back(axis_name);
      children.Set(parent_name, std::move(value));
    } else {
      Array<String> value{axis_name};
      children.Set(parent_name, std::move(value));
    }
  }
  node->parent = std::move(parent);
  node->children = std::move(children);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(AxisTreeNode);

TVM_REGISTER_GLOBAL("tir.sparse.AxisTree")
    .set_body_typed([](Array<String> axis_names, Array<Optional<String>> axis_parent_names) {
      return AxisTree(axis_names, axis_parent_names);
    });

/******** SparseBuffer ********/

/*! \brief Default constructor of SparseBuffer */
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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SparseBufferNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SparseBufferNode*>(node.get());
      p->stream << "sparse_buffer(" << op->name << ", [";
      for (int i = 0, n = static_cast<int>(op->axes.size()); i < n; ++i) {
        const Axis& axis = op->axes[i];
        p->stream << axis;
        if (i < n - 1) {
          p->stream << ", ";
        }
      }
      p->stream << "], " << op->data << ")";
    });

/******** AxisKind ********/

/*! \brief Printer function of Axiskind. */
std::ostream& operator<<(std::ostream& out, AxisKind type) {
  switch (type) {
    case AxisKind::kDenseFixed:
      out << "dense-fixed";
      break;
    case AxisKind::kDenseVariable:
      out << "dense-variable";
      break;
    case AxisKind::kSparseFixed:
      out << "sparse-fixed";
      break;
    case AxisKind::kSparseVariable:
      out << "sparse-variable";
      break;
    default:
      LOG(FATAL) << "Cannot reach here";
  }
  return out;
}

/******** SpIterVar ********/

/*! \brief Default constructor of SpIterVar. */
SpIterVar::SpIterVar(Var var, PrimExpr max_extent, bool is_reduction, Axis axis) {
  ObjectPtr<SpIterVarNode> node = make_object<SpIterVarNode>();

  arith::Analyzer ana;

  node->var = Var(std::move(var));
  node->max_extent = std::move(max_extent);
  node->is_reduction = is_reduction;
  node->axis = std::move(axis);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(SpIterVarNode);

TVM_REGISTER_GLOBAL("tir.sparse.SpIterVar")
    .set_body_typed([](Var var, PrimExpr max_extent, bool is_reduction, Axis axis) {
      return SpIterVar(var, max_extent, is_reduction, axis);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SpIterVarNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SpIterVarNode*>(node.get());
      p->stream << "sp_iter_var(" << op->var->name_hint << ", " << op->max_extent << ", "
                << (op->is_reduction ? "reduction" : "spatial") << ", " << op->axis->name << ")";
    });

}  // namespace tir
}  // namespace tvm
