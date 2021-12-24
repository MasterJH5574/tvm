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
      return;
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
PrimExpr AggregateOffset(PrimExpr prev_offset, Axis axis, PrimExpr index, arith::Analyzer* ana) {
  PrimExpr new_offset;
  switch (axis->kind()) {
    case AxisKind::kDenseFixed: {
      new_offset = std::move(prev_offset) * axis->length + std::move(index);
      break;
    }
    case AxisKind::kSparseFixed: {
      auto sf_axis = axis.as<SparseFixedAxisNode>();
      new_offset = std::move(prev_offset) * sf_axis->nnz_cols + std::move(index);
      break;
    }
    case AxisKind::kDenseVariable: {
      // TODO(zihao): finish the aggregating offset for attached axis.
      auto dv_axis = axis.as<DenseVariableAxisNode>();
      new_offset = add(BufferLoad(dv_axis->indptr, {std::move(prev_offset)}), std::move(index));
      break;
    }
    case AxisKind::kSparseVariable: {
      auto sv_axis = axis.as<SparseVariableAxisNode>();
      new_offset = add(BufferLoad(sv_axis->indptr, {std::move(prev_offset)}), std::move(index));
      break;
    }
  }
  return ana->Simplify(new_offset);
}

/*! \brief A class storing the context information of sparse blocks. */
class SparseBlockCtx {
 private:
  struct Scope {
    explicit Scope(SparseBlock sp_block, arith::Analyzer* ana)
        : sp_block(std::move(sp_block)), ana(ana) {
      for (const SpIterVar& sp_iter_var : this->sp_block->sp_iter_vars) {
        axis2sp_iter.Set(sp_iter_var->axis, sp_iter_var);
        sp_iter_var_map.Set(sp_iter_var->var, sp_iter_var);
      }
    }

    /*! \brief The sparse block */
    SparseBlock sp_block;
    /*! \brief A mapping from axes to the sparse iterators that go over them */
    Map<Axis, SpIterVar> axis2sp_iter;
    /*! \brief A mapping from the internal variables of sparse iterators to the iterators */
    Map<Var, SpIterVar> sp_iter_var_map;
    /*! \brief The stored offsets of the sparse iterators in the sparse block */
    Map<SpIterVar, PrimExpr> cached_offsets;
    /*! \brief The analyzer */
    arith::Analyzer* ana;
  };

 public:
  explicit SparseBlockCtx(arith::Analyzer* ana) : ana_(ana) {}

  void EnterScope(const SparseBlockNode* sp_block) {
    stack_.emplace_back(GetRef<SparseBlock>(sp_block), ana_);
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

  /*!
   * \brief Get the parent sparse iterator of the input sparse iterator, i.e., the sparse iterator
   * of the axis that the input depends on.
   * \param sp_iter The sparse iterator whose parent iterator is to be looked up.
   * \return The parent sparse iterator of the input, or `NullOpt` if the input has no parent.
   */
  Optional<SpIterVar> GetParentSpIterVar(const SpIterVar& sp_iter) {
    const Optional<Axis>& parent_axis = sp_iter->axis->GetParentAxis();
    if (parent_axis.defined()) {
      return top()->axis2sp_iter.Get(parent_axis.value());
    } else {
      return NullOpt;
    }
  }

  /*!
   * \brief Get the offset of the input sparse iterator according to the iterators it depends on.
   * \param sp_iter The sparse iterator to be queried.
   * \return The offset of the sparse iterator.
   */
  PrimExpr GetOffset(const SpIterVar& sp_iter) {
    const Optional<PrimExpr>& offset = top()->cached_offsets.Get(sp_iter);
    if (offset.defined()) {
      return offset.value();
    } else {
      const Optional<SpIterVar>& parent_sp_iter = GetParentSpIterVar(sp_iter);
      PrimExpr prev_ofs = parent_sp_iter.defined() ? GetOffset(parent_sp_iter.value()) : Integer(0);
      PrimExpr new_ofs = AggregateOffset(std::move(prev_ofs), sp_iter->axis, sp_iter->var, ana_);
      top()->cached_offsets.Set(sp_iter, new_ofs);
      return new_ofs;
    }
  }

  /*!
   * \brief Get the coordinate of the input sparse iterator in the current scope.
   * \param sp_iter The sparse iterator to be queried.
   * \return The coordinate of the sparse iterator.
   */
  PrimExpr GetCoordinate(SpIterVar sp_iter) {
    const Axis& axis = sp_iter->axis;
    AxisKind kind = axis->kind();
    if (kind == AxisKind::kDenseFixed || kind == AxisKind::kDenseVariable) {
      // if dense, just return the value.
      return sp_iter->var;
    }

    PrimExpr offset = GetOffset(sp_iter);
    if (kind == AxisKind::kSparseFixed) {
      return BufferLoad(Downcast<SparseFixedAxis>(axis)->indices, {std::move(offset)});
    } else {  // AxisKind::kSparseVariable
      return BufferLoad(Downcast<SparseVariableAxis>(axis)->indices, {std::move(offset)});
    }
  }

  /*!
   * \brief Get the iteration extent of the input sparse iterator.
   * \param sp_iter_var The sparse iterator to be queried.
   * \return The iteration extent of the input sparse iterator.
   */
  PrimExpr GetIterExtent(SpIterVar sp_iter) {
    Optional<SpIterVar> parent_sp_iter = GetParentSpIterVar(sp_iter);
    PrimExpr prev_off = parent_sp_iter.defined() ? GetOffset(parent_sp_iter.value()) : Integer(0);
    return ana_->Simplify(sub(AggregateOffset(add(prev_off, 1), sp_iter->axis, Integer(0), ana_),
                              AggregateOffset(prev_off, sp_iter->axis, Integer(0), ana_)));
  }

 private:
  std::vector<Scope> stack_;
  arith::Analyzer* ana_;

  /*! \brief The top scope in the sparse block stack. */
  inline Scope* top() const { return const_cast<Scope*>(&stack_.back()); }
};

/*! \brief A class storing the context information of sparse buffer accesses. */
class SparseBufferAccessCtx {
 private:
  struct Scope {
    explicit Scope(SparseBuffer sp_buffer, const SparseBlockCtx* sp_blk_ctx, arith::Analyzer* ana)
        : sp_buffer(std::move(sp_buffer)), sp_blk_ctx(sp_blk_ctx), ana(ana) {
      offsets.reserve(this->sp_buffer->ndim() + 1);
      offsets.push_back(Integer(0));
      n_matched_iters = 0;
    }

    /*! \brief The sparse buffer */
    SparseBuffer sp_buffer;
    /*! \brief The accumulative offsets at each dimension in this access */
    Array<PrimExpr> offsets;
    /*! \brief The number of iterators starting from the beginning that match the axes of the
     * sparse buffer */
    int n_matched_iters;
    /*! \brief The sparse block context of the sparse block that this access is in */
    const SparseBlockCtx* sp_blk_ctx;
    /*! \brief The analyzer */
    arith::Analyzer* ana;
  };

 public:
  explicit SparseBufferAccessCtx(arith::Analyzer* ana) : ana_(ana) {}

  void EnterScope(SparseBuffer sp_buffer, const SparseBlockCtx* sp_blk_ctx) {
    stack_.emplace_back(sp_buffer, sp_blk_ctx, ana_);
  }

  void ExitScope() { stack_.pop_back(); }

  /*!
   * \brief Get the axis of the sparse buffer at the input dimension.
   * \param dim The dimension to be queried.
   * \return The axis of the sparse buffer at the specific dimension.
   */
  inline Axis GetAxis(int dim) const { return top()->sp_buffer->axes[dim]; }

  /*!
   * \brief Check whether the iterators from the first dimension to the specific dimension match the
   * axes of the sparse buffer.
   * \param dim The dimension to be checked.
   * \return Whether iterators match.
   */
  inline bool MatchWithSpBlock(int dim) const { return top()->n_matched_iters > dim; }

  /*!
   * \brief Get the range of the indices of the access at the specific dimension.
   * \param dim The dimension to be checked.
   * \return The range of the indices of the buffer access.
   */
  std::tuple<PrimExpr, PrimExpr> GetIndicesRange(int dim) {
    const PrimExpr& prev_ofs = top()->offsets[dim];
    const Axis& axis = top()->sp_buffer->axes[dim];
    return {AggregateOffset(prev_ofs, axis, Integer(0), ana_),
            AggregateOffset(add(prev_ofs, 1), axis, Integer(0), ana_)};
  }

  /*!
   * \brief Update the number of matched iterators.
   * \param index The index used to update.
   * \param dim The dimension where the index is.
   */
  void UpdateMatch(const PrimExpr& index, int dim) {
    ICHECK(dim + 1 == static_cast<int>(top()->offsets.size()));
    const Axis& axis = GetAxis(dim);

    if (const auto* var = index.as<VarNode>()) {
      Optional<SpIterVar> sp_iter = top()->sp_blk_ctx->GetSparseIterVar(var);
      if (sp_iter.defined() && sp_iter.value()->axis.same_as(axis)) {
        ++(top()->n_matched_iters);
      }
    }
  }

  /*!
   * \brief Update the accumulative offset of this buffer access.
   * \param index The index used to update.
   * \param dim The dimension where the index is.
   * \return The offset at the input dimension.
   */
  PrimExpr UpdateOffset(PrimExpr index, int dim) {
    const Axis& axis = GetAxis(dim);
    PrimExpr new_offset = AggregateOffset(top()->offsets.back(), axis, std::move(index), ana_);
    top()->offsets.push_back(new_offset);
    return new_offset;
  }

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
   * \brief Return the viewed index of the input index at the given dimension, i.e., return the
   * coordinate of the index for dense axes, and the position of the index for sparse axes.
   * \param index The input index.
   * \param dim The dimension where the index is.
   * \return The viewed index of the input index.
   */
  PrimExpr ViewIndexInAxis(PrimExpr index, int dim) {
    // Update the number of matched sparse iterators.
    // If the index at this dimension matches, just return the index.
    sp_buf_ctx_.UpdateMatch(index, dim);
    if (sp_buf_ctx_.MatchWithSpBlock(dim)) {
      return index;
    }

    // Iteratively transform the index to coordinate.
    PrimExpr coordinate = ExprMutator::VisitExpr(index);
    const Axis& axis = sp_buf_ctx_.GetAxis(dim);
    if (axis->kind() == AxisKind::kDenseFixed || axis->kind() == AxisKind::kDenseVariable) {
      // Return the coordinate of the index for dense axes.
      return coordinate;
    } else {
      // Return the position of the index for sparse axes by binary search on the `indices` buffer.
      PrimExpr l, r;
      std::tie(l, r) = sp_buf_ctx_.GetIndicesRange(dim);
      Buffer indices = axis->kind() == AxisKind::kSparseFixed
                           ? Downcast<SparseFixedAxis>(axis)->indices
                           : Downcast<SparseVariableAxis>(axis)->indices;
      return lower_bound(indices->data, coordinate, l, r) - l;
    }
  }

  /*!
   * \brief Compute the offset of given access indices with regard to the given sparse buffer.
   * \param sp_buffer The sparse buffer to be accessed.
   * \param indices The access indices.
   */
  PrimExpr ComputeOffset(SparseBuffer sp_buffer, Array<PrimExpr> indices) {
    ICHECK_EQ(static_cast<int>(indices.size()), sp_buffer->ndim());

    PrimExpr offset = Integer(0);
    for (int i = 0; i < sp_buffer->ndim(); ++i) {
      offset = sp_buf_ctx_.UpdateOffset(ViewIndexInAxis(indices[i], i), i);
    }
    return offset;
  }

  /*!
   * \brief Convert the input sparse iterator to a block iterator.
   * \param sp_iter The sparse iterator to be converted.
   * \param var_map The mapping from sparse iterators to loop variables, for extent substitution.
   * \return The corresponding block iterator.
   */
  IterVar SpIterVarToIterVar(SpIterVar sp_iter, const Map<Var, PrimExpr>& var_map) {
    // Substitute the iteration vars in the expression with the loop vars.
    return IterVar(Range::FromMinExtent(0, Substitute(sp_blk_ctx_.GetIterExtent(sp_iter), var_map)),
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
    Optional<SpIterVar> sp_iter = sp_blk_ctx_.GetSparseIterVar(var);
    return sp_iter.defined() ? sp_blk_ctx_.GetCoordinate(sp_iter.value()) : GetRef<PrimExpr>(var);
  }

  PrimExpr VisitExpr_(const SparseBufferLoadNode* load) final {
    buffer_read_.insert(load->buffer.get());
    sp_buf_ctx_.EnterScope(load->buffer, &sp_blk_ctx_);
    PrimExpr offset = ComputeOffset(load->buffer, load->indices);
    sp_buf_ctx_.ExitScope();
    return BufferLoad(load->buffer->data, {std::move(offset)});
  }

  Stmt VisitStmt_(const SparseBufferStoreNode* store) final {
    buffer_write_.insert(store->buffer.get());
    PrimExpr value = ExprMutator::VisitExpr(store->value);
    sp_buf_ctx_.EnterScope(store->buffer, &sp_blk_ctx_);
    PrimExpr offset = ComputeOffset(store->buffer, store->indices);
    sp_buf_ctx_.ExitScope();
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
      bool NeedCreateNewBlock(const Axis& axis, const Map<Axis, Var>& axis2loop_var,
                              const std::unordered_set<const VarNode*>& defined_loop_vars) {
        if (axis->kind() != AxisKind::kDenseVariable && axis->kind() != AxisKind::kSparseVariable) {
          return false;
        }

        Optional<Var> loop_var = axis2loop_var.Get(axis->GetParentAxis().value());
        CHECK(loop_var.defined()) << "ValueError: The parent axis of " << axis
                                  << "does not appear in the sparse block";

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
