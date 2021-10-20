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
 * \brief tvm/tir/sparse.h
 * \brief sparse axes and buffers.
 */
#ifndef TVM_TIR_SPARSE_H_
#define TVM_TIR_SPARSE_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/var.h>

namespace tvm {
namespace tir {

enum class AxisKind : int {
  kDenseFixed = 0,
  kDenseVariable = 1,
  kSparseFixed = 2,
  kSparseVariable = 3
};

class Axis;

/*! \brief Common interface for both SparseBlockCtx and SparseBufferAccessCtx. */
class SparseCtx {
 public:
  virtual Optional<Axis> GetPrevAxis(Axis axis) const = 0;
  virtual PrimExpr GetCoordinate(Axis axis) const = 0;
  virtual PrimExpr GetOffset(Axis axis) const = 0;
  virtual void SetCoordinate(Axis axis, PrimExpr coordinate) = 0;
  virtual void SetOffset(Axis axis, PrimExpr index) = 0;
};

/*!
 * \brief Base type for axis in sparse formats.
 */
class AxisNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("length", &length);
  }

  bool SEqualReduce(const AxisNode* other, SEqualReducer equal) const {
    return equal(name, other->name) && equal(length, other->length);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
    hash_reduce(length);
  }

  /* name of current axis. */
  String name;
  /* length of current axis. For sparse axis, length refers to the upperbound of
   * the current axis. */
  PrimExpr length;

  String GetName() const { return name; }
  PrimExpr GetLength() const { return length; }
  DataType GetIndexType() const { return length->dtype; }
  virtual Optional<Axis> GetParentAxis() const = 0;

  virtual AxisKind kind() const = 0;
  virtual PrimExpr GetNNZ() const = 0;

  virtual PrimExpr Aggregate(SparseCtx* ctx, PrimExpr index) const = 0;
  virtual PrimExpr Compress(SparseCtx* ctx, PrimExpr coordinate) const = 0;
  virtual PrimExpr Decompress(SparseCtx* ctx, PrimExpr offset, PrimExpr index) const = 0;
  std::tuple<PrimExpr, PrimExpr> GetOffsetExtent(SparseCtx* ctx) const;

  static constexpr const char* _type_key = "tir.sparse.Axis";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(AxisNode, Object);
};

/*!
 * \brief Managed reference to AxisNode.
 * \sa AxisNode
 */
class Axis : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Axis, ObjectRef, AxisNode);
};

/*!
 * \brief Dense axis whose column indices are consecutive.
 */
class DenseAxisNode : public AxisNode {
 public:
  static constexpr const char* _type_key = "tir.sparse.DenseAxis";
  TVM_DECLARE_BASE_OBJECT_INFO(DenseAxisNode, AxisNode);
};

/*!
 * \brief Managed reference to DenseAxisNode.
 * \sa DenseAxisNode
 */
class DenseAxis : public Axis {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(DenseAxis, Axis, DenseAxisNode);
};

/*!
 * \brief Sparse axis whose column indices is not consecutive.
 */
class SparseAxisNode : public AxisNode {
 public:
  static constexpr const char* _type_key = "tir.sparse.SparseAxis";
  TVM_DECLARE_BASE_OBJECT_INFO(SparseAxisNode, AxisNode);
};

/*!
 * \brief Managed reference to SparseAxisNode.
 * \sa SparseAxisNode
 */
class SparseAxis : public Axis {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(SparseAxis, Axis, SparseAxisNode);
};

/*!
 * \brief Dense axis with fixed length per row.
 */
class DenseFixedAxisNode : public DenseAxisNode {
 public:
  AxisKind kind() const final { return AxisKind::kDenseFixed; }

  PrimExpr GetNNZ() const final { return length; }

  Optional<Axis> GetParentAxis() const final { return NullOpt; }

  PrimExpr Aggregate(SparseCtx* ctx, PrimExpr index) const;

  PrimExpr Compress(SparseCtx* ctx, PrimExpr coordinate) const;

  PrimExpr Decompress(SparseCtx* ctx, PrimExpr offset, PrimExpr index) const;

  static constexpr const char* _type_key = "tir.sparse.DenseFixedAxis";
  TVM_DECLARE_BASE_OBJECT_INFO(DenseFixedAxisNode, DenseAxisNode);
};

/*!
 * \brief Managed reference to DenseFixedAxisNode.
 * \sa DenseFixedAxisNode
 */
class DenseFixedAxis : public DenseAxis {
 public:
  TVM_DLL explicit DenseFixedAxis(String name, PrimExpr length);

  TVM_DEFINE_OBJECT_REF_METHODS(DenseFixedAxis, DenseAxis, DenseFixedAxisNode);
};

/*! \brief Derivation axis, constructed by T.dense(axis). */
class DenseFromSparseAxisNode : public DenseFixedAxisNode {
 public:
  void VisitAttrs(AttrVisitor* v) {
    DenseFixedAxisNode::VisitAttrs(v);
    v->Visit("base", &base);
  }

  bool SEqualReduce(const DenseFromSparseAxisNode* other, SEqualReducer equal) const {
    return DenseFixedAxisNode::SEqualReduce(other, equal) && equal(base, other->base);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    DenseFixedAxisNode::SHashReduce(hash_reduce);
    hash_reduce(base);
  }

  /* The based sparse axis. */
  SparseAxis base;

  static constexpr const char* _type_key = "tir.sparse.DenseFromSparseAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(DenseFromSparseAxisNode, DenseFixedAxisNode);
};

/*!
 * \brief Managed reference of DenseFromSparseAxisNode.
 * \sa DenseFromSparseAxisNode
 */
class DenseFromSparseAxis : public DenseFixedAxis {
 public:
  /* DenseFromSparseAxis could be constructed by specifying the based sparse axis. */
  TVM_DLL explicit DenseFromSparseAxis(SparseAxis base);

  TVM_DEFINE_OBJECT_REF_METHODS(DenseFromSparseAxis, DenseFixedAxis, DenseFromSparseAxisNode);
};

class FusedAxis;

/*! \brief Derivation axis, constructed by T.fuse(axis1, axis2, ...) */
class FusedAxisNode : public DenseFixedAxisNode {
 public:
  /* The group of axes to be fused. */
  Array<Axis> group;
  /* The index of current FusedAxis in the group. */
  int index;

  void VisitAttrs(AttrVisitor* v) {
    DenseFixedAxisNode::VisitAttrs(v);
    v->Visit("group", &group);
    v->Visit("index", &index);
  }

  bool SEqualReduce(const FusedAxisNode* other, SEqualReducer equal) const {
    return DenseFixedAxisNode::SEqualReduce(other, equal) && equal(group, other->group) &&
           equal(index, other->index);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    DenseFixedAxisNode::SHashReduce(hash_reduce);
    hash_reduce(group);
    hash_reduce(index);
  }

  static constexpr const char* _type_key = "tir.sparse.FusedAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(FusedAxisNode, DenseFixedAxisNode);
};

/*!
 * \brief Managed reference to FusedAxisNode.
 * \sa FusedAxisNode
 */
class FusedAxis : public DenseFixedAxis {
 public:
  /* Fused axis could be constructed by specifying a group of based axes and an index */
  TVM_DLL explicit FusedAxis(Array<Axis> group, int index);

  TVM_DEFINE_OBJECT_REF_METHODS(FusedAxis, DenseFixedAxis, FusedAxisNode);
};

/*!
 * \brief Dense axis with variable length, such as ragged tensor.
 */
class DenseVariableAxisNode : public DenseAxisNode {
 public:
  Buffer indptr;
  PrimExpr nnz_;
  Axis parent_;

  void VisitAttrs(AttrVisitor* v) {
    DenseAxisNode::VisitAttrs(v);
    v->Visit("indptr", &indptr);
  }

  bool SEqualReduce(const DenseVariableAxisNode* other, SEqualReducer equal) const {
    return DenseAxisNode::SEqualReduce(other, equal) && equal(indptr, other->indptr);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    DenseAxisNode::SHashReduce(hash_reduce);
    hash_reduce(indptr);
  }

  AxisKind kind() const final { return AxisKind::kDenseVariable; }

  PrimExpr GetNNZ() const final { return nnz_; }

  Optional<Axis> GetParentAxis() const final { return parent_; }

  PrimExpr Aggregate(SparseCtx* ctx, PrimExpr index) const;

  PrimExpr Compress(SparseCtx* ctx, PrimExpr coordinate) const;

  PrimExpr Decompress(SparseCtx* ctx, PrimExpr offset, PrimExpr index) const;

  static constexpr const char* _type_key = "tir.sparse.DenseVariableAxis";
  TVM_DECLARE_BASE_OBJECT_INFO(DenseVariableAxisNode, DenseAxisNode);
};

/*!
 * \brief Managed reference to DenseVariableAxisNode.
 * \sa DenseVariableAxisNode
 */
class DenseVariableAxis : public DenseAxis {
 public:
  TVM_DLL explicit DenseVariableAxis(String name, Axis parent, PrimExpr length, PrimExpr nnz,
                                     Buffer indptr);

  TVM_DEFINE_OBJECT_REF_METHODS(DenseVariableAxis, DenseAxis, DenseVariableAxisNode);
};

/*!
 * \brief Dense variable axis attached to another dense variable axis.
 */
class AttachedAxisNode : public DenseVariableAxisNode {
 public:
  /* The original axis before attaching. */
  DenseVariableAxis orig_;

  PrimExpr Aggregate(SparseCtx* ctx, PrimExpr index) const;

  static constexpr const char* _type_key = "tir.sparse.AttachedAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttachedAxisNode, DenseVariableAxisNode);
};

/*!
 * \brief Managed reference to AttachedAxisNode.
 * \sa AttachedAxisNode
 */
class AttachedAxis : public DenseVariableAxis {
 public:
  TVM_DLL explicit AttachedAxis(String name, Axis parent, DenseVariableAxis orig, PrimExpr nnz,
                                Buffer indptr);
  TVM_DEFINE_OBJECT_REF_METHODS(AttachedAxis, DenseVariableAxis, AttachedAxisNode);
};

/*!
 * \brief Sparse axis with fixed number of non-zero columns per row.
 */
class SparseFixedAxisNode : public SparseAxisNode {
 public:
  Buffer indices;
  Axis parent_;
  /* fixed number of non-zero columns of current sparse axis. */
  PrimExpr nnz_cols;

  void VisitAttrs(AttrVisitor* v) {
    SparseAxisNode::VisitAttrs(v);
    v->Visit("indptr", &indices);
    v->Visit("nnz_cols", &nnz_cols);
  }

  bool SEqualReduce(const SparseFixedAxisNode* other, SEqualReducer equal) const {
    return SparseAxisNode::SEqualReduce(other, equal) && equal(indices, other->indices) &&
           equal(nnz_cols, other->nnz_cols);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    SparseAxisNode::SHashReduce(hash_reduce);
    hash_reduce(indices);
    hash_reduce(nnz_cols);
  }

  PrimExpr GetNNZ() const { return indices->shape[0]; }

  AxisKind kind() const final { return AxisKind::kSparseFixed; }

  Optional<Axis> GetParentAxis() const final { return parent_; }

  PrimExpr Aggregate(SparseCtx* ctx, PrimExpr index) const;

  PrimExpr Compress(SparseCtx* ctx, PrimExpr coordinate) const;

  PrimExpr Decompress(SparseCtx* ctx, PrimExpr offset, PrimExpr index) const;

  static constexpr const char* _type_key = "tir.sparse.SparseFixedAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(SparseFixedAxisNode, SparseAxisNode);
};

/*!
 * \brief Managed reference to SparseFixedAxisNode.
 * \sa SparseFixedAxisNode
 */
class SparseFixedAxis : public SparseAxis {
 public:
  TVM_DLL explicit SparseFixedAxis(String name, Axis parent, PrimExpr length, Buffer indices,
                                   PrimExpr nnz_cols);

  TVM_DEFINE_OBJECT_REF_METHODS(SparseFixedAxis, SparseAxis, SparseFixedAxisNode);
};

/*!
 * \brief Sparse axis with variable number of non-zero columns per row.
 */
class SparseVariableAxisNode : public SparseAxisNode {
 public:
  Buffer indptr;
  Buffer indices;
  Axis parent_;

  void VisitAttrs(AttrVisitor* v) {
    SparseAxisNode::VisitAttrs(v);
    v->Visit("indptr", &indptr);
    v->Visit("indices", &indices);
  }

  bool SEqualReduce(const SparseVariableAxisNode* other, SEqualReducer equal) const {
    return SparseAxisNode::SEqualReduce(other, equal) && equal(indptr, other->indptr) &&
           equal(indices, other->indices);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    SparseAxisNode::SHashReduce(hash_reduce);
    hash_reduce(indptr);
    hash_reduce(indices);
  }

  PrimExpr GetNNZ() const { return indices->shape[0]; }

  AxisKind kind() const final { return AxisKind::kSparseVariable; }

  Optional<Axis> GetParentAxis() const final { return parent_; }

  PrimExpr Aggregate(SparseCtx* ctx, PrimExpr index) const;

  PrimExpr Compress(SparseCtx* ctx, PrimExpr coordinate) const;

  PrimExpr Decompress(SparseCtx* ctx, PrimExpr offset, PrimExpr index) const;

  static constexpr const char* _type_key = "tir.sparse.SparseVariableAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(SparseVariableAxisNode, SparseAxisNode);
};

/*!
 * \brief Managed reference to SparseVariableAxisNode.
 * \sa SparseVariableAxisNode
 */
class SparseVariableAxis : public SparseAxis {
 public:
  TVM_DLL explicit SparseVariableAxis(String name, Axis parent, PrimExpr length, Buffer indptr,
                                      Buffer indices);

  TVM_DEFINE_OBJECT_REF_METHODS(SparseVariableAxis, SparseAxis, SparseVariableAxisNode);
};

/*!
 * \brief Class of sparse buffer.
 */
class SparseBufferNode : public Object {
 public:
  /* Axes */
  Array<Axis> axes;
  /* Buffer corresponding to flattened value */
  Buffer data;
  /* Buffer Name */
  String name;

  inline int ndim() const { return static_cast<int>(axes.size()); }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("axes", &axes);
    v->Visit("data", &data);
    v->Visit("name", &name);
  }

  bool SEqualReduce(const SparseBufferNode* other, SEqualReducer equal) const {
    return equal(axes, other->axes) && equal(data, other->data) && equal(name, other->name);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(axes);
    hash_reduce(data);
    hash_reduce(name);
  }

  static constexpr const char* _type_key = "tir.sparse.SparseBuffer";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(SparseBufferNode, Object);
};

/*!
 * \brief Managed reference to SparseBufferNode.
 * \sa SparseBufferNode
 */
class SparseBuffer : public ObjectRef {
 public:
  TVM_DLL explicit SparseBuffer(Array<Axis> axes, Buffer data, String name);

  TVM_DEFINE_OBJECT_REF_METHODS(SparseBuffer, ObjectRef, SparseBufferNode);
};

// overload printing of for type.
TVM_DLL std::ostream& operator<<(std::ostream& os, AxisKind kind);

/*!
 * \brief Iterator variables in SparseTIR
 */
class SpIterVarNode : public Object {
 public:
  Var var;
  PrimExpr max_extent;
  bool is_reduction;
  Axis axis;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("var", &var);
    v->Visit("max_extent", &max_extent);
    v->Visit("axis", &axis);
    v->Visit("is_reduction", &is_reduction);
  }

  bool SEqualReduce(const SpIterVarNode* other, SEqualReducer equal) const {
    return equal(var, other->var) && equal(max_extent, other->max_extent) &&
           equal(axis, other->axis) && equal(is_reduction, other->is_reduction);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(var);
    hash_reduce(max_extent);
    hash_reduce(axis);
    hash_reduce(is_reduction);
  }

  static constexpr const char* _type_key = "tir.sparse.SpIterVar";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(SpIterVarNode, Object);
};

class SpIterVar : public ObjectRef {
 public:
  TVM_DLL explicit SpIterVar(Var var, PrimExpr max_extent, bool is_reduction, Axis axis);

  /*!
   * \return the corresponding var in the IterVar.
   */
  inline operator PrimExpr() const;

  TVM_DEFINE_OBJECT_REF_METHODS(SpIterVar, ObjectRef, SpIterVarNode);
};

// inline implementations
inline SpIterVar::operator PrimExpr() const { return (*this)->var; }

// inline implementations
inline const char* SpIterKind2String(AxisKind t) {
  switch (t) {
    case AxisKind::kDenseFixed:
      return "dense_fixed";
    case AxisKind::kDenseVariable:
      return "dense_variable";
    case AxisKind::kSparseFixed:
      return "sparse_fixed";
    case AxisKind::kSparseVariable:
      return "sparse_variable";
  }
  LOG(FATAL) << "Unknown AxisKind" << t;
  throw;
}

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SPARSE_H_
