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
#include <tvm/tir/op.h>
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

TVM_REGISTER_GLOBAL("tir.sparse.GetNNZ").set_body_typed([](Axis axis) { return axis->GetNNZ(); });

TVM_REGISTER_GLOBAL("tir.sparse.GetParent").set_body_typed([](Axis axis) { return axis->GetParentAxis(); });

/******** AxisNode ********/

std::tuple<PrimExpr, PrimExpr> AxisNode::GetOffsetExtent(SparseCtx* ctx) const {
  auto prev = ctx->GetPrevAxis(GetRef<Axis>(this));
  if (prev.defined()) {
    Axis prev_axis = prev.value();
    PrimExpr lb = Aggregate(ctx, 0);
    PrimExpr orig_prev_coordinate = ctx->GetCoordinate(prev_axis),
             orig_prev_offset = ctx->GetOffset(prev_axis);
    ctx->SetCoordinate(prev_axis, orig_prev_coordinate + 1);
    ctx->SetOffset(prev_axis, orig_prev_offset + 1);
    PrimExpr ub = Aggregate(ctx, 0);
    ctx->SetCoordinate(prev_axis, orig_prev_coordinate);
    ctx->SetOffset(prev_axis, orig_prev_offset);
    return {lb, ub};
  } else {
    return {Integer(0), GetNNZ()};
  }
};

/******** DenseFixedAxis ********/

/*! \brief Default constructor of DenseFixedAxis */
DenseFixedAxis::DenseFixedAxis(String name, PrimExpr length) {
  ObjectPtr<DenseFixedAxisNode> node = make_object<DenseFixedAxisNode>();
  node->name = std::move(name);
  node->length = std::move(length);
  data_ = std::move(node);
}

PrimExpr DenseFixedAxisNode::Aggregate(SparseCtx* ctx, PrimExpr index) const {
  auto try_prev = ctx->GetPrevAxis(GetRef<Axis>(this));
  if (try_prev.defined()) {
    Axis prev_axis = try_prev.value();
    PrimExpr prev_offset = ctx->GetOffset(prev_axis);
    return prev_offset * length + std::move(index);
  } else {
    return index;
  }
}

PrimExpr DenseFixedAxisNode::Compress(SparseCtx* ctx, PrimExpr coordinate) const {
  return coordinate;
}

PrimExpr DenseFixedAxisNode::Decompress(SparseCtx* ctx, PrimExpr offset, PrimExpr index) const {
  return index;
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

/******** DenseFromSparseAxis ********/

/*! \brief Default constructor of DenseFromSparseAxis */
DenseFromSparseAxis::DenseFromSparseAxis(SparseAxis base) {
  ObjectPtr<DenseFromSparseAxisNode> node = make_object<DenseFromSparseAxisNode>();
  node->name = base->name + "_dense";
  node->length = base->length;
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
  for (size_t i = 1; i < group.size(); ++i) {
    fused_name += group[i]->name;
  }
  node->name = "fused_" + fused_name + "_" + group[index]->name;
  node->length = group[index]->GetNNZ();
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

/******** DenseVariableAxis ********/

/*! \brief Default constructor of DenseVariableAxis */
DenseVariableAxis::DenseVariableAxis(String name, Axis parent, PrimExpr length, PrimExpr nnz,
                                     Buffer indptr) {
  ObjectPtr<DenseVariableAxisNode> node = make_object<DenseVariableAxisNode>();
  node->name = std::move(name);
  node->parent_ = std::move(parent);
  node->length = std::move(length);
  node->nnz_ = std::move(nnz);
  node->indptr = std::move(indptr);
  data_ = std::move(node);
}

PrimExpr DenseVariableAxisNode::Aggregate(SparseCtx* ctx, PrimExpr index) const {
  Axis prev_axis = ctx->GetPrevAxis(GetRef<Axis>(this)).value();
  PrimExpr prev_offset = ctx->GetOffset(prev_axis);
  return BufferLoad(indptr, {std::move(prev_offset)}) + std::move(index);
}

PrimExpr DenseVariableAxisNode::Compress(SparseCtx* ctx, PrimExpr coordinate) const {
  return coordinate;
}

PrimExpr DenseVariableAxisNode::Decompress(SparseCtx* ctx, PrimExpr offset, PrimExpr index) const {
  return index;
}

TVM_REGISTER_NODE_TYPE(DenseVariableAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.DenseVariableAxis")
    .set_body_typed([](String name, Axis parent, PrimExpr length, PrimExpr nnz, Buffer indptr) {
      return DenseVariableAxis(std::move(name), std::move(parent), std::move(length),
                               std::move(nnz), std::move(indptr));
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DenseVariableAxisNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const DenseVariableAxisNode*>(node.get());
      p->stream << "dense_variable(" << op->name << ", " << op->length << ", " << op->indptr->name
                << ")";
    });

/******** AttachedAxis ********/
/*! \brief Default constructor of AttachedAxis */
AttachedAxis::AttachedAxis(String name, Axis parent, DenseVariableAxis orig, PrimExpr nnz,
                           Buffer indptr) {
  ObjectPtr<AttachedAxisNode> node = make_object<AttachedAxisNode>();
  node->name = std::move(name);
  node->parent_ = std::move(parent);
  node->orig_ = std::move(orig);
  node->length = node->orig_->length;
  node->nnz_ = std::move(nnz);
  node->indptr = std::move(indptr);
  data_ = std::move(node);
}

PrimExpr AttachedAxisNode::Aggregate(SparseCtx* ctx, PrimExpr index) const {
  PrimExpr root_offset = ctx->GetOffset(orig_->parent_);
  PrimExpr accum_offset = BufferLoad(indptr, {root_offset});
  Array<DenseVariableAxis> collect_axes;
  Array<PrimExpr> collect_coordinates;
  Array<PrimExpr> strides;
  Axis axis;
  PrimExpr stride = Integer(1);
  for (axis = GetRef<Axis>(this); axis->kind() == AxisKind::kDenseVariable;
       axis = ctx->GetPrevAxis(axis).value()) {
    DenseVariableAxis dv_axis = Downcast<DenseVariableAxis>(axis);
    collect_axes.push_back(dv_axis);
    collect_coordinates.push_back(ctx->GetCoordinate(axis));
    Buffer indptr;
    if (auto att_axis = dv_axis.as<AttachedAxisNode>()) {
      indptr = att_axis->orig_->indptr;
    } else {
      indptr = dv_axis->indptr;
    }
    strides.push_back(stride);
    stride = stride * (BufferLoad(indptr, {root_offset + 1}) - BufferLoad(indptr, {root_offset}));
  }
  ICHECK(axis == orig_->parent_) << "Root axis mismatch.";
  PrimExpr length = Integer(0);
  for (int i = collect_axes.size() - 1; i >= 0; --i) {
    DenseVariableAxis axis = std::move(collect_axes[i]);
    PrimExpr coordinate = std::move(collect_coordinates[i]);
    PrimExpr stride = std::move(strides[i]);
    accum_offset = accum_offset + coordinate * stride;
  }
  return accum_offset;
}

TVM_REGISTER_NODE_TYPE(AttachedAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.AttachedAxis")
    .set_body_typed([](String name, Axis parent, DenseVariableAxis orig, PrimExpr nnz,
                       Buffer indptr) {
      return AttachedAxis(std::move(name), std::move(parent), std::move(orig), std::move(nnz),
                          std::move(indptr));
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AttachedAxisNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AttachedAxisNode*>(node.get());
      p->stream << "attached_axis(" << op->name << ", " << op->length << ", " << op->indptr->name
                << ")";
    });

/******** SparseFixedAxis ********/

/*! \brief Default constructor of SparseFixedAxis */
SparseFixedAxis::SparseFixedAxis(String name, Axis parent, PrimExpr length, Buffer indices,
                                 PrimExpr nnz_cols) {
  ObjectPtr<SparseFixedAxisNode> node = make_object<SparseFixedAxisNode>();
  node->name = std::move(name);
  node->parent_ = std::move(parent);
  node->length = std::move(length);
  node->indices = std::move(indices);
  node->nnz_cols = std::move(nnz_cols);
  data_ = std::move(node);
}

PrimExpr SparseFixedAxisNode::Aggregate(SparseCtx* ctx, PrimExpr index) const {
  Axis prev_axis = ctx->GetPrevAxis(GetRef<Axis>(this)).value();
  PrimExpr prev_offset = ctx->GetOffset(prev_axis);
  return std::move(prev_offset) * nnz_cols + std::move(index);
}

PrimExpr SparseFixedAxisNode::Compress(SparseCtx* ctx, PrimExpr coordinate) const {
  PrimExpr lb, ub;
  std::tie(lb, ub) = GetOffsetExtent(ctx);
  return lower_bound(indices->data, coordinate, lb, ub) - lb;
}

PrimExpr SparseFixedAxisNode::Decompress(SparseCtx* ctx, PrimExpr offset, PrimExpr index) const {
  return BufferLoad(indices, {offset});
}

TVM_REGISTER_NODE_TYPE(SparseFixedAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.SparseFixedAxis")
    .set_body_typed([](String name, Axis parent, PrimExpr length, Buffer indices,
                       PrimExpr nnz_cols) {
      return SparseFixedAxis(std::move(name), std::move(parent), std::move(length),
                             std::move(indices), std::move(nnz_cols));
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SparseFixedAxisNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SparseFixedAxisNode*>(node.get());
      p->stream << "sparse_fixed(" << op->name << ", " << op->parent_->name << ", " << op->length
                << ", " << op->nnz_cols << ", " << op->indices->name << ")";
    });

/******** SparseVariableAxis ********/

/*! \brief Default constructor of SparseVariableAxis */
SparseVariableAxis::SparseVariableAxis(String name, Axis parent, PrimExpr length, Buffer indptr,
                                       Buffer indices) {
  ObjectPtr<SparseVariableAxisNode> node = make_object<SparseVariableAxisNode>();
  node->name = std::move(name);
  node->parent_ = std::move(parent);
  node->length = std::move(length);
  node->indptr = std::move(indptr);
  node->indices = std::move(indices);
  data_ = std::move(node);
}

PrimExpr SparseVariableAxisNode::Aggregate(SparseCtx* ctx, PrimExpr index) const {
  Axis prev_axis = ctx->GetPrevAxis(GetRef<Axis>(this)).value();
  PrimExpr prev_offset = ctx->GetOffset(prev_axis);
  return BufferLoad(indptr, {std::move(prev_offset)}) + std::move(index);
}

PrimExpr SparseVariableAxisNode::Compress(SparseCtx* ctx, PrimExpr coordinate) const {
  PrimExpr lb, ub;
  std::tie(lb, ub) = GetOffsetExtent(ctx);
  return lower_bound(indices->data, coordinate, lb, ub) - lb;
}

PrimExpr SparseVariableAxisNode::Decompress(SparseCtx* ctx, PrimExpr offset, PrimExpr index) const {
  return BufferLoad(indices, {offset});
}

TVM_REGISTER_NODE_TYPE(SparseVariableAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.SparseVariableAxis")
    .set_body_typed([](String name, Axis parent, PrimExpr length, Buffer indptr, Buffer indices) {
      return SparseVariableAxis(std::move(name), std::move(parent), std::move(length),
                                std::move(indptr), std::move(indices));
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SparseVariableAxisNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SparseVariableAxisNode*>(node.get());
      p->stream << "sparse_variable(" << op->name << ", " << op->length << ", " << op->indptr->name
                << ", " << op->indices->name << ")";
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
      return SparseBuffer(std::move(axes), std::move(data), std::move(name));
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
      return SpIterVar(std::move(var), std::move(max_extent), is_reduction, std::move(axis));
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SpIterVarNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SpIterVarNode*>(node.get());
      p->stream << "sp_iter_var(" << op->var->name_hint << ", " << op->max_extent << ", "
                << (op->is_reduction ? "reduction" : "spatial") << ", " << op->axis->name << ")";
    });

}  // namespace tir
}  // namespace tvm
