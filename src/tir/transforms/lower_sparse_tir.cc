/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file lower_sparse_tir.cc
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <utility>

#include "ir_utils.h"

namespace tvm {
namespace tir {

class SparseTIRLowerer : public StmtExprMutator {
 private:
  std::pair<Buffer, PrimExpr> LowerIndices(SparseBuffer sp_buffer, Array<PrimExpr> indices) {
    int ndim = sp_buffer->ndim();
    ICHECK_EQ(static_cast<int>(indices.size()), ndim);
    PrimExpr lowered_index = Integer(0);

    for (int i = 0; i < ndim; ++i) {
      const Axis& axis = sp_buffer->axes[i];
      const PrimExpr& index = indices[i];

      // Stage 1.
      PrimExpr sp_index{nullptr};
      if (const auto* sp_iter = index.as<SpIterVarNode>()) {
        SpIterKind kind = sp_iter->kind;
        if (kind == SpIterKind::kDenseFixed) {
          CHECK(!axis->IsInstance<DenseVariableAxisNode>());
          if (const auto* df_axis = axis.as<DenseFixedAxisNode>()) {
            CHECK(ana.CanProveEqual(sp_iter->max_extent, df_axis->length));
            sp_index = GetRef<SpIterVar>(sp_iter);
          } else {
            PrimExpr l = LowerIndex(lowered_index, sp_buffer, i, 0);
            PrimExpr r = LowerIndex(Add(lowered_index, 1), sp_buffer, i, 0);
            Var buffer_var;
            if (const auto* sf_axis = axis.as<SparseFixedAxisNode>()) {
              CHECK(ana.CanProveEqual(sp_iter->max_extent, sf_axis->length));
              buffer_var = sf_axis->indices->data;
            } else if (const auto* sv_axis = axis.as<SparseVariableAxisNode>()) {
              CHECK(ana.CanProveEqual(sp_iter->max_extent, sv_axis->length));
              buffer_var = sv_axis->indices->data;
            } else {
              LOG(FATAL) << "Cannot reach here";
            }
            sp_index = lower_bound(buffer_var, index, std::move(l), std::move(r));
          }
        } else if (kind == SpIterKind::kDenseVariable) {
          const auto* dv_axis = axis.as<DenseVariableAxisNode>();
          CHECK(dv_axis != nullptr);
          CHECK(sp_iter->axis.defined());
          sp_index = GetRef<SpIterVar>(sp_iter);
        } else if (kind == SpIterKind::kSparseFixed) {
          CHECK(!axis->IsInstance<DenseVariableAxisNode>());
          CHECK(sp_iter->axis.defined());
          const Axis& iterated_axis = sp_iter->axis.value();
          if (const auto* df_axis = axis.as<DenseFixedAxisNode>()) {
            CHECK(ana.CanProveEqual(sp_iter->max_extent, df_axis->length));
            // Todo: convert to dense
          } else if (const auto* sf_axis = axis.as<SparseFixedAxisNode>()) {
            CHECK(ana.CanProveEqual(sp_iter->max_extent, sf_axis->length));
            if (iterated_axis.get() == sf_axis) {
              sp_index = GetRef<SpIterVar>(sp_iter);
            } else {
              // Todo: convert to dense and do binary search
            }
          } else if (const auto* sv_axis = axis.as<SparseVariableAxisNode>()) {
            CHECK(ana.CanProveEqual(sp_iter->max_extent, sv_axis->length));
            // Todo: convert to dense and do binary search
          } else {
            LOG(FATAL) << "Cannot reach here";
          }
        } else {
          CHECK(kind == SpIterKind::kSparseVariable);
          CHECK(!axis->IsInstance<DenseVariableAxisNode>());
          CHECK(sp_iter->axis.defined());
          const Axis& iterated_axis = sp_iter->axis.value();
          if (const auto* df_axis = axis.as<DenseFixedAxisNode>()) {
            CHECK(ana.CanProveEqual(sp_iter->max_extent, df_axis->length));
            // Todo: convert to dense
          } else if (const auto* sf_axis = axis.as<SparseFixedAxisNode>()) {
            CHECK(ana.CanProveEqual(sp_iter->max_extent, sf_axis->length));
            // Todo: convert to dense and do binary search
          } else if (const auto* sv_axis = axis.as<SparseVariableAxisNode>()) {
            CHECK(ana.CanProveEqual(sp_iter->max_extent, sv_axis->length));
            if (iterated_axis.get() == sv_axis) {
              sp_index = GetRef<SpIterVar>(sp_iter);
            } else {
              // Todo: convert to dense and do binary search
            }
          } else {
            LOG(FATAL) << "Cannot reach here";
          }
        }
      } else {
        // Todo
      }

      // Stage 2.
      lowered_index = LowerIndex(std::move(lowered_index), sp_buffer, i, sp_index);
    }

    return std::make_pair(sp_buffer->data, lowered_index);
  }

  PrimExpr LowerIndex(PrimExpr prev_lowered_index, SparseBuffer sp_buffer, int dim,
                      PrimExpr index) {
    const Axis& axis = sp_buffer->axes[dim];
    if (axis->IsInstance<DenseFixedAxisNode>() || axis->IsInstance<SparseFixedAxisNode>()) {
      return ana.Simplify(prev_lowered_index * axis->length + index);
    } else if (const auto* dv_axis = axis.as<DenseVariableAxisNode>()) {
      return ana.Simplify(Add(BufferLoad(dv_axis->indptr, {prev_lowered_index}), index));
    } else if (const auto* sv_axis = axis.as<SparseVariableAxisNode>()) {
      return ana.Simplify(Add(BufferLoad(sv_axis->indptr, {prev_lowered_index}), index));
    }
    LOG(FATAL) << "Cannot reach here";
    throw;
  }

  PrimExpr VisitExpr_(const SparseBufferLoadNode* load) final {
    std::pair<Buffer, PrimExpr> res = LowerIndices(load->buffer, load->indices);
    return BufferLoad(std::move(res.first), {std::move(res.second)});
  }

  Stmt VisitStmt_(const SparseBufferStoreNode* store) final {
    PrimExpr value = ExprMutator::VisitExpr(store->value);
    std::pair<Buffer, PrimExpr> res = LowerIndices(store->buffer, store->indices);
    return BufferStore(std::move(res.first), std::move(value), {std::move(res.second)});
  }

  arith::Analyzer ana;
};

PrimFunc LowerSparseTIR(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = SparseTIRLowerer()(std::move(f->body));
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass LowerSparseTIR() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerSparseTIR(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerSparseTIR", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerSparseTIR").set_body_typed(LowerSparseTIR);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
