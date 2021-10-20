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

#include <set>
#include <utility>

#include "../../support/utils.h"
#include "../schedule/analysis.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Add the buffers accessed in sparse blocks to the PrimFunc's buffer map.
 * \param f The PrimFunc whose buffer map is to be updated.
 * \return The up to date buffer map.
 */
Map<Var, Buffer> UpdateBufferMap(PrimFunc f) {
  struct BufferMapUpdater : public StmtVisitor {
    explicit BufferMapUpdater(Map<Var, Buffer> buffer_map) : buffer_map_(std::move(buffer_map)) {}

    void VisitStmt_(const SparseBlockNode* sp_block) {
      for (const auto& it : sp_block->sp_struct_param_map) {
        const ObjectRef& sp_struct = it.first;
        const Array<Var>& params = it.second;
        if (const auto* dv_axis = sp_struct.as<DenseVariableAxisNode>()) {
          // collect indptr buffer of dense variable axis.
          ICHECK_EQ(params.size(), 1);
          buffer_map_.Set(params[0], dv_axis->indptr);
        } else if (const auto* sf_axis = sp_struct.as<SparseFixedAxisNode>()) {
          // collect indices buffer of sparse fixed axis.
          ICHECK_EQ(params.size(), 1);
          buffer_map_.Set(params[0], sf_axis->indices);
        } else if (const auto* sv_axis = sp_struct.as<SparseVariableAxisNode>()) {
          // collect indptr and indices buffer of sparse variable axis.
          ICHECK_EQ(params.size(), 2);
          buffer_map_.Set(params[0], sv_axis->indptr);
          buffer_map_.Set(params[1], sv_axis->indices);
        } else if (const auto* sp_buffer = sp_struct.as<SparseBufferNode>()) {
          // collect data buffer for sparse buffers.
          ICHECK_EQ(params.size(), 1);
          buffer_map_.Set(params[0], sp_buffer->data);
        }
      }
    }

    Map<Var, Buffer> buffer_map_;
  };

  BufferMapUpdater updater(f->buffer_map);
  updater(f->body);
  return std::move(updater.buffer_map_);
}

/*!
 * \brief Aggregate the offset on previous axes with the index on the current axis.
 * \param prev_offset The lowered offset accumulated over all prior axes.
 * \param axis The current axis.
 * \param index The sparse index on current axis.
 * \param ana The analyzer used for expression simplification.
 * \return The aggregated offset.
 */
PrimExpr AggregateOffset(SparseCtx* ctx, Axis axis, PrimExpr index, arith::Analyzer* ana) {
  PrimExpr new_offset = axis->Aggregate(ctx, index);
  if (ana != nullptr) {
    return ana->Simplify(new_offset);
  } else {
    return new_offset;
  }
}

/*! \brief A class storing the context information of sparse blocks. */
class SparseBlockCtx : public SparseCtx {
 private:
  struct Scope {
    explicit Scope(SparseBlock sp_block) : sp_block(std::move(sp_block)) {
      for (const SpIterVar& sp_iter_var : this->sp_block->sp_iter_vars) {
        sp_iter_var_map.Set(sp_iter_var->var, sp_iter_var);
      }
    }

    /*! \brief The sparse block */
    SparseBlock sp_block;
    /*! \brief A mapping from the internal variables of sparse iterators to the iterators */
    Map<Var, SpIterVar> sp_iter_var_map;
    /*! \brief The stored offsets of the axis in the sparse block */
    Map<Axis, PrimExpr> cached_offsets;
    /*! \brief The stored coordinates of the axis in the sparse block */
    Map<Axis, PrimExpr> cached_coordinates;
  };

 public:
  explicit SparseBlockCtx(arith::Analyzer* ana) : ana_(ana) {}

  void EnterScope(const SparseBlockNode* sp_block) {
    stack_.emplace_back(GetRef<SparseBlock>(sp_block));
    /* Compute offsets and coordinates */
    size_t n_iters = sp_block->sp_iter_vars.size();
    for (size_t i = 0; i < n_iters;) {
      SpIterVar sp_iter_var = sp_block->sp_iter_vars[i];
      Axis axis = sp_iter_var->axis;

      PrimExpr offset, index;
      if (auto fused_axis = axis.as<FusedAxisNode>()) {
        auto group = fused_axis->group;
        offset = sp_block->sp_iter_vars[i + group.size() - 1]->var;
        for (int j = group.size() - 1; j >= 0; --j) {
          Axis orig = group[j];
          SetOffset(orig, offset);
          if (j > 0) {
            Buffer indptr;
            if (auto sv_axis = orig.as<SparseVariableAxisNode>()) {
              indptr = sv_axis->indptr;
            } else if (auto dv_axis = orig.as<DenseVariableAxisNode>()) {
              indptr = dv_axis->indptr;
            } else {
              throw;
            }
            offset = upper_bound(indptr->data, offset, Integer(0), indptr->shape[0]) - 1;
          }
        }
        for (size_t j = 0; j < group.size(); ++j) {
          Axis orig = group[j];
          offset = GetOffset(orig);
          PrimExpr lb = std::get<0>(orig->GetOffsetExtent(this));
          index = offset - lb;
          PrimExpr coordinate = orig->Decompress(this, offset, index);
          SetCoordinate(orig, coordinate);
          i++;
        }
      } else {
        offset = AggregateOffset(this, axis, sp_iter_var->var, ana_);
        index = sp_iter_var->var;
        PrimExpr coordinate = axis->Decompress(this, offset, index);
        SetOffset(axis, offset);
        SetCoordinate(axis, coordinate);
        i++;
      }
    }
  }

  void ExitScope() { stack_.pop_back(); }

  /*!
   * \brief Get the sparse iterator corresponding to the given variable in the current scope.
   * \param var The variable whose corresponding sparse iterator is to be looked up.
   * \return The corresponding sparse iterator of the input variable, or `NullOpt` if the input
   * variable does not corresponds to a sparse iterator.
   */
  Optional<SpIterVar> GetSparseIterVar(const VarNode* var) const {
    return top()->sp_iter_var_map.Get(GetRef<Var>(var));
  }

  Optional<Axis> GetPrevAxis(Axis axis) const {
    // In Sparse block, previous axis is parent axis.
    return axis->GetParentAxis();
  }

  void SetOffset(Axis axis, PrimExpr offset) { top()->cached_offsets.Set(axis, offset); }

  void SetCoordinate(Axis axis, PrimExpr idx) { top()->cached_coordinates.Set(axis, idx); }

  /*!
   * \brief Get the offset of the input axis in the block.
   * \param sp_iter The axis to be queried.
   * \return The offset.
   */
  PrimExpr GetOffset(Axis axis) const {
    Optional<PrimExpr> try_offset = top()->cached_offsets.Get(axis);
    CHECK(try_offset.defined()) << "The offset of axis " << axis->name << " not defined yet.";
    PrimExpr offset = try_offset.value();
    return std::move(offset);
  }

  /*!
   * \brief Get the coordinate of the input axis in the block.
   * \param axis The axis to be queried.
   * \return The coordinate.
   */
  PrimExpr GetCoordinate(Axis axis) const {
    Optional<PrimExpr> try_index = top()->cached_coordinates.Get(axis);
    CHECK(try_index.defined()) << "The index of axis not defined yet.";
    PrimExpr index = try_index.value();
    return std::move(index);
  }

  /*!
   * \brief Get the iteration extent of the input sparse iterator.
   * \param sp_iter_var The sparse iterator to be queried.
   * \return The iteration extent of the input sparse iterator.
   */
  PrimExpr GetIterExtent(SpIterVar sp_iter) {
    if (const auto* fused_axis = sp_iter->axis.as<FusedAxisNode>()) {
      // Fused axis.
      if (fused_axis->index == int(fused_axis->group.size() - 1)) {
        // The last axis in the fused group.
        return fused_axis->GetNNZ();
      } else {
        return Integer(1);
      }
    }
    PrimExpr lb, ub;
    std::tie(lb, ub) = sp_iter->axis->GetOffsetExtent(this);
    return ana_->Simplify(ub - lb);
  }

  Optional<Axis> MatchAxis(SparseCtx* buf_ctx, Axis axis) {
    if (!top()->cached_offsets.Get(axis).defined()) {
      return NullOpt;
    } else {
      Axis axis_ = axis;
      auto prev = buf_ctx->GetPrevAxis(axis);
      auto blk_prev = GetPrevAxis(axis);
      for (; prev.defined();) {
        if (prev != blk_prev) {
          return NullOpt;
        } else {
          axis_ = prev.value();
          prev = buf_ctx->GetPrevAxis(axis_);
          blk_prev = GetPrevAxis(axis_);
        }
      }
      return axis;
    }
  }

  bool MatchIndex(Optional<Axis> matched_axis, PrimExpr expr) {
    if (!matched_axis.defined()) {
      return false;
    }
    auto var = expr.as<VarNode>();
    if (var == nullptr) {
      return false;
    }
    auto try_sp_iter_var = top()->sp_iter_var_map.Get(GetRef<Var>(var));
    if (!try_sp_iter_var.defined()) {
      return false;
    }
    Axis axis = try_sp_iter_var.value()->axis;
    if (auto fused_axis = axis.as<FusedAxisNode>()) {
      axis = fused_axis->group[fused_axis->index];
    }
    return axis == matched_axis.value();
  }

 private:
  std::vector<Scope> stack_;
  arith::Analyzer* ana_;

  /*! \brief The top scope in the sparse block stack. */
  inline Scope* top() const { return const_cast<Scope*>(&stack_.back()); }
};

/*! \brief A class storing the context information of sparse buffer accesses. */
class SparseBufferAccessCtx : public SparseCtx {
 private:
  struct Scope {
    explicit Scope(Array<Axis> axes) : axes(std::move(axes)) {}

    Array<Axis> axes;
    /*! \brief The stored offsets of the axis in the sparse buffer */
    Map<Axis, PrimExpr> cached_offsets;
    /*! \brief The stored coordinates of the axis in the sparse buffer */
    Map<Axis, PrimExpr> cached_coordinates;
    PrimExpr final_offset;
  };

 public:
  explicit SparseBufferAccessCtx(arith::Analyzer* ana) : ana_(ana) {}

  void EnterScope(SparseBuffer sp_buffer, Array<PrimExpr> raw_indices_, Array<PrimExpr> coordinates,
                  SparseBlockCtx* sp_blk_ctx) {
    stack_.emplace_back(sp_buffer->axes);
    size_t n_dims = sp_buffer->axes.size();
    ICHECK(n_dims == raw_indices_.size())
        << "The number of indices does not equal number of axes in the sparse buffer.";
    ICHECK(n_dims == coordinates.size())
        << "The number of coordinates does not equal number of axes in the sparse buffer.";

    /* Compute offsets and coordinates. */
    for (size_t i = 0; i < n_dims; ++i) {
      Axis axis = sp_buffer->axes[i];
      PrimExpr coordinate = coordinates[i];
      SetCoordinate(axis, coordinate);
      auto try_parent = axis->GetParentAxis();
      // update axis match

      auto matched_axis = sp_blk_ctx->MatchAxis(this, axis);
      // compute offset
      PrimExpr offset = (sp_blk_ctx->MatchIndex(matched_axis, raw_indices_[i]))
                            ? sp_blk_ctx->GetOffset(axis)
                            : AggregateOffset(this, axis, axis->Compress(this, coordinate), ana_);
      SetOffset(axis, offset);
      if (i + 1 == n_dims) {
        // the final axis;
        top()->final_offset = offset;
      }
    }
  }

  void ExitScope() { stack_.pop_back(); }

  Optional<Axis> GetPrevAxis(Axis axis) const {
    Array<Axis> axes = top()->axes;
    Optional<Axis> ret = NullOpt;
    for (auto it : axes) {
      if (it == axis) {
        break;
      }
      ret = it;
    }
    return ret;
  }

  void SetOffset(Axis axis, PrimExpr offset) { top()->cached_offsets.Set(axis, offset); }

  void SetCoordinate(Axis axis, PrimExpr coordinate) {
    top()->cached_coordinates.Set(axis, coordinate);
  }

  PrimExpr GetOffset(Axis axis) const {
    auto try_offset = top()->cached_offsets.Get(axis);
    CHECK(try_offset.defined()) << "The offset of the axis is not defined.";
    PrimExpr offset = try_offset.value();
    return std::move(offset);
  }

  PrimExpr GetCoordinate(Axis axis) const {
    auto try_coordinate = top()->cached_coordinates.Get(axis);
    CHECK(try_coordinate.defined()) << "The coordinate of the axis is not defined.";
    PrimExpr coordinate = try_coordinate.value();
    return std::move(coordinate);
  }

  PrimExpr GetLastOffset() const { return top()->final_offset; }

 private:
  std::vector<Scope> stack_;
  arith::Analyzer* ana_;

  /*! \brief The top scope in the sparse buffer access stack. */
  inline Scope* top() const { return const_cast<Scope*>(&stack_.back()); }
};

/*!
 * \brief Rewrite the high-dimensional sparse buffers and access indices to low-level buffers and
 * offsets.
 */
class IndexTransformer : public StmtExprMutator {
 public:
  explicit IndexTransformer() : sp_blk_ctx_(&ana_), sp_buf_ctx_(&ana_) {}

 private:
  /*!
   * \brief Convert the input sparse iterator to a block iterator.
   * \param sp_iter The sparse iterator to be converted.
   * \param var_map The mapping from sparse iterators to loop variables, for extent substitution.
   * \return The corresponding block iterator.
   */
  IterVar SpIterVarToIterVar(const SpIterVar& sp_iter, Map<Var, PrimExpr> var_map) {
    // Substitute the iteration vars in the expression with the loop vars.
    return IterVar(Range::FromMinExtent(0, sp_blk_ctx_.GetIterExtent(sp_iter)),
                   sp_iter->var, sp_iter->is_reduction ? kCommReduce : kDataPar);
  }

  /*!
   * \brief Generate the read and write regions for sparse blocks.
   * \param sp_block The sparse block, which is the source of the reads and writes.
   * \param reads The read regions of the sparse block.
   * \param writes The write regions of the sparse block.
   */
  void GenerateReadWriteRegions(const SparseBlockNode* sp_block, Array<BufferRegion>* reads,
                                Array<BufferRegion>* writes) {
    for (const ObjectRef& obj : sp_block->sp_structs) {
      if (const auto* dv_axis = obj.as<DenseVariableAxisNode>()) {
        reads->push_back(BufferRegion::FullRegion(dv_axis->indptr));
      } else if (const auto* sf_axis = obj.as<SparseFixedAxisNode>()) {
        reads->push_back(BufferRegion::FullRegion(sf_axis->indices));
      } else if (const auto* sv_axis = obj.as<SparseVariableAxisNode>()) {
        reads->push_back(BufferRegion::FullRegion(sv_axis->indptr));
        reads->push_back(BufferRegion::FullRegion(sv_axis->indices));
      } else if (const auto* sp_buffer = obj.as<SparseBufferNode>()) {
        if (buffer_read_.count(sp_buffer)) {
          reads->push_back(BufferRegion::FullRegion(sp_buffer->data));
        }
        if (buffer_write_.count(sp_buffer)) {
          writes->push_back(BufferRegion::FullRegion(sp_buffer->data));
        }
      }
    }
  }

  /*!
   * \brief Generated the loop nests for the outside the input body.
   * \param body The statement to be wrapped by loop nests.
   * \param block_iters The block iterators defined in the outermost block in `body`.
   * \param loop_vars The loop variables of the loops to be generated.
   * \return The outermost generated loop.
   */
  Stmt GenerateLoops(Stmt body, const Array<IterVar>& block_iters, const Array<Var>& loop_vars) {
    int n_iter = static_cast<int>(block_iters.size());
    for (int i = n_iter - 1; i >= 0; --i) {
      const Range& dom = block_iters[i]->dom;
      body = For(loop_vars[i], dom->min, dom->extent, ForKind::kSerial, std::move(body));
    }
    return body;
  }

  PrimExpr VisitExpr_(const VarNode* var) final {
    auto try_sp_iter = sp_blk_ctx_.GetSparseIterVar(var);
    if (try_sp_iter.defined()) {
      SpIterVar sp_iter = try_sp_iter.value();
      Axis axis = sp_iter->axis;
      if (auto fused_axis = axis.as<FusedAxisNode>()) {
        axis = fused_axis->group[fused_axis->index];
      }
      return sp_blk_ctx_.GetCoordinate(axis);
    } else {
      return GetRef<PrimExpr>(var);
    }
  }

  PrimExpr VisitExpr_(const SparseBufferLoadNode* load) final {
    buffer_read_.insert(load->buffer.get());
    Array<PrimExpr> coordinates;
    for (const PrimExpr& index : load->indices) {
      coordinates.push_back(VisitExpr(index));
    }
    sp_buf_ctx_.EnterScope(load->buffer, load->indices, coordinates, &sp_blk_ctx_);
    PrimExpr offset = sp_buf_ctx_.GetLastOffset();
    sp_buf_ctx_.ExitScope();
    return BufferLoad(load->buffer->data, {std::move(offset)});
  }

  Stmt VisitStmt_(const SparseBufferStoreNode* store) final {
    buffer_write_.insert(store->buffer.get());
    Array<PrimExpr> coordinates;
    for (const PrimExpr& index : store->indices) {
      coordinates.push_back(VisitExpr(index));
    }
    sp_buf_ctx_.EnterScope(store->buffer, store->indices, coordinates, &sp_blk_ctx_);
    PrimExpr offset = sp_buf_ctx_.GetLastOffset();
    sp_buf_ctx_.ExitScope();
    PrimExpr value = VisitExpr(store->value);
    return BufferStore(store->buffer->data, std::move(value), {std::move(offset)});
  }

  Stmt VisitStmt_(const SparseBlockNode* sp_block) final {
    /*! \brief A class temporarily storing the block signatures and the outer loop variables of the
     * blocks to be generated */
    struct BlockInfo {
      /*! \brief The outer loop variables of the block */
      Array<Var> loop_vars;
      /*! \brief The block iterators of the block */
      Array<IterVar> block_iters;
      /*! \brief The block iterator bindings of the block */
      Array<PrimExpr> iter_bindings;
      /*! \brief The init statement of the block */
      Optional<Stmt> init;

      /*!
       * \brief Push a new loop variable/block iterator/iterator binding to this block.
       * \param loop_var The loop variable to be pushed.
       * \param block_iter The block iterator to be pushed.
       * \param iter_binding The iterator binding to be pushed.
       */
      void Push(const Var& loop_var, const IterVar& block_iter, const PrimExpr& iter_binding) {
        loop_vars.push_back(loop_var);
        block_iters.push_back(block_iter);
        iter_bindings.push_back(iter_binding);
      }

      /*!
       * \brief Check whether the input loop variable exists in the outer loop variables of this
       * block.
       * \param target_loop_var The loop variable to be checked
       * \return Whether the input loop variable exists in the outer loop variables of this block.
       */
      bool LoopVarAppears(const Var& target_loop_var) {
        for (const Var& loop_var : loop_vars) {
          if (loop_var.same_as(target_loop_var)) {
            return true;
          }
        }
        return false;
      }

      /*!
       * \brief Check whether a new block is needed. We need to create a new block when:
       * - the input axis is variable (dense-variable or sparse-variable), and
       * - the parent axis of the input axis has corresponding loop variable in the current block.
       * \param axis The axis to be checked.
       * \param axis2loop_var The mapping from axes to their corresponding loop variables.
       * \param defined_loop_vars The loop variables defined in previous blocks
       * (excluding the current one).
       * \return Whether a new block is needed according to the conditions above.
       */
      bool NeedCreateNewBlock(Axis axis, Map<Axis, Var> axis2loop_var,
                              const std::unordered_set<const VarNode*>& defined_loop_vars) {
        if (axis->kind() != AxisKind::kDenseVariable && axis->kind() != AxisKind::kSparseVariable) {
          return false;
        }

        const Optional<Var>& loop_var = axis2loop_var.Get(axis->GetParentAxis().value());
        CHECK(loop_var.defined()) << "ValueError: The parent axis of " << axis
                                  << " does not appear in the sparse block";

        if (LoopVarAppears(loop_var.value())) {
          return true;
        }
        CHECK(defined_loop_vars.count(loop_var.value().get()))
            << "ValueError: The parent axis of " << axis
            << " should appear before it in the sparse block";
        return false;
      }
    };

    int n_iter = static_cast<int>(sp_block->sp_iter_vars.size());
    buffer_read_.clear();
    buffer_write_.clear();

    // Step 1. Enter a new sparse block scope.
    sp_blk_ctx_.EnterScope(sp_block);

    // Step 2. Recursively mutate the `init` field and the block body.
    Optional<Stmt> init =
        sp_block->init.defined() ? VisitStmt(sp_block->init.value()) : Optional<Stmt>(NullOpt);
    Stmt body = VisitStmt(sp_block->body);

    // Step 3. Create the new loop variables.
    Map<Var, PrimExpr> var_map;
    Map<Axis, Var> axis2loop_var;
    for (const SpIterVar& sp_iter_var : sp_block->sp_iter_vars) {
      Var loop_var("v_" + sp_iter_var->var->name_hint);
      var_map.Set(sp_iter_var->var, loop_var);
      if (auto fused_axis = sp_iter_var->axis.as<FusedAxisNode>()) {
        // handle the special case of fused_axis
        axis2loop_var.Set(fused_axis->group[fused_axis->index], loop_var);
      }
      axis2loop_var.Set(sp_iter_var->axis, loop_var);
    }

    // Step 4. Gather the information of the blocks to be generated.
    std::unordered_set<const VarNode*> defined_loop_vars;
    std::vector<BlockInfo> block_infos(1);
    /* Whether a reduction block iterator has appeared */
    bool has_reduction_var = false;

    for (int i = 0; i < n_iter; ++i) {
      SpIterVar sp_it_var = sp_block->sp_iter_vars[i];
      if (block_infos.back().NeedCreateNewBlock(sp_it_var->axis, axis2loop_var,
                                                defined_loop_vars)) {
        // Mark the loop variables corresponding to the current block as "defined".
        for (const Var& loop_var : block_infos.back().loop_vars) {
          defined_loop_vars.insert(loop_var.get());
        }
        // Create a new BlockInfo.
        block_infos.emplace_back();
      }

      Var loop_var = Downcast<Var>(var_map.Get(sp_it_var->var));
      block_infos.back().Push(loop_var, SpIterVarToIterVar(sp_it_var, var_map), loop_var);
      if (!has_reduction_var && sp_it_var->is_reduction) {
        block_infos.back().init = std::move(init);
        has_reduction_var = true;
      }
    }

    // Step 5. Generate the read-regions and write-retions of the block.
    Array<BufferRegion> reads;
    Array<BufferRegion> writes;
    GenerateReadWriteRegions(sp_block, &reads, &writes);

    // Step 6. Generate nested blocks and loops from innermost to outermost.
    for (int i = static_cast<int>(block_infos.size()) - 1; i >= 0; --i) {
      BlockInfo info = std::move(block_infos[i]);
      Block block(/*iter_vars=*/info.block_iters,
                  /*reads=*/reads,
                  /*writes=*/writes,
                  /*name_hint=*/sp_block->name + std::to_string(i),
                  /*body=*/std::move(body),
                  /*init=*/std::move(info.init),
                  /*alloc_buffers=*/{},
                  /*match_buffers=*/{},
                  /*annotations=*/{{"sparse", Bool(true)}});
      BlockRealize block_realize(/*iter_values=*/std::move(info.iter_bindings),
                                 /*predicate=*/const_true(),
                                 /*block=*/std::move(block));
      Stmt loop = GenerateLoops(std::move(block_realize), std::move(info.block_iters),
                                std::move(info.loop_vars));
      body = std::move(loop);
    }

    // Step 7: Exit the sparse block scope.
    sp_blk_ctx_.ExitScope();

    return body;
  }

  SparseBlockCtx sp_blk_ctx_;
  SparseBufferAccessCtx sp_buf_ctx_;
  std::unordered_set<const SparseBufferNode*> buffer_read_;
  std::unordered_set<const SparseBufferNode*> buffer_write_;
  arith::Analyzer ana_;
};

/*!
 * \brief Wrap the body statement with an empty root block.
 * \param body The body statements to wrap with.
 * \return The wrapped block.
 */
Stmt WrapWithRootBlock(Stmt body) {
  Block root_block({}, {}, {}, "root", std::move(body));
  return BlockRealize({}, const_true(), std::move(root_block));
}

PrimFunc LowerSparseTIR(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    // Step 1. Update the PrimFunc's buffer map.
    fptr->buffer_map = UpdateBufferMap(f);
    // Step 2. Lower indices.
    fptr->body = IndexTransformer()(std::move(fptr->body));
    // Step 3. Wrap the function body with a root block.
    fptr->body = WrapWithRootBlock(std::move(fptr->body));
    return f;
  } else {
    return f;
  }
}

namespace transform {

/*!
 * \brief The lowering pass from TIR to Sparse TIR.
 */
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
