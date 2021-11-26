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
#include <stack>
#include <utility>

#include "../schedule/analysis.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Get the mapping from Var to corresponding Buffer's.
 * \param f The primitive function to visit.
 * \return The map.
 */
Map<Var, Buffer> UpdateBufferMap(PrimFunc f) {
  struct BufferMapUpdater : public StmtVisitor {
    explicit BufferMapUpdater(Map<Var, Buffer> buffer_map) : buffer_map_(std::move(buffer_map)) {}

    /*!
     * \brief Visit function to collect var to buffer mapping in a sparse block.
     * \param sp_block The sparse block to collect.
     */
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
 * \brief Rewrite indices in sparse buffers to indices in corresponding data
 * buffers.
 */
class IndexTransformer : public StmtExprMutator {
 public:
  explicit IndexTransformer(AccessAndDependencyCollector collector, AxisTree axis_tree)
      : collector_(std::move(collector)), axis_tree_(std::move(axis_tree)) {}

 private:
  /*!
   * \brief The lowered absolute offset of an sparse buffer access pattern.
   * \param sp_buffer The sparse buffer to be accessed.
   * \param indices The sparse indices to access the buffer.
   * \return The lowered absolute offset to the start of flattened data in given sparse buffer.
   */
  PrimExpr LowerIndices(SparseBuffer sp_buffer, const Array<PrimExpr>& indices) {
    int ndim = sp_buffer->ndim();
    int n_lower = static_cast<int>(indices.size());
    ICHECK_LE(n_lower, ndim);

    PrimExpr lowered_index = Integer(0);

    for (int i = 0; i < n_lower; ++i) {
      const Axis& axis = sp_buffer->axes[i];
      const PrimExpr& index = indices[i];

      // Stage 1. Get the sparse index.
      SpIterVar sp_iter = collector_.GetSpIterFromIndex(index);
      PrimExpr sp_index{nullptr};

      PrimExpr l = PartialLowerIndex(lowered_index, sp_buffer->axes[i], 0);
      PrimExpr r = PartialLowerIndex(add(lowered_index, 1), sp_buffer->axes[i], 0);

      switch (sp_iter->kind) {
        case SpIterKind::kDenseFixed: {
          CHECK(!axis->IsInstance<DenseVariableAxisNode>());
          if (const auto* df_axis = axis.as<DenseFixedAxisNode>()) {
            CHECK(ana_.CanProveEqual(sp_iter->max_extent, df_axis->length));
            sp_index = sp_iter;
          } else {
            Var buffer_var;
            if (const auto* sf_axis = axis.as<SparseFixedAxisNode>()) {
              CHECK(ana_.CanProveEqual(sp_iter->max_extent, sf_axis->length));
              buffer_var = sf_axis->indices->data;
            } else if (const auto* sv_axis = axis.as<SparseVariableAxisNode>()) {
              CHECK(ana_.CanProveEqual(sp_iter->max_extent, sv_axis->length));
              buffer_var = sv_axis->indices->data;
            } else {
              LOG(FATAL) << "Cannot reach here";
            }
            sp_index = lower_bound(buffer_var, index, std::move(l), std::move(r));
          }
          break;
        }
        case SpIterKind::kDenseVariable: {
          const auto* dv_axis = axis.as<DenseVariableAxisNode>();
          CHECK(dv_axis != nullptr);
          CHECK(sp_iter->axis.defined());
          sp_index = sp_iter;
          break;
        }
        case SpIterKind::kSparseFixed: {
          CHECK(!axis->IsInstance<DenseVariableAxisNode>());
          CHECK(sp_iter->axis.defined());
          const Axis& iterated_axis = sp_iter->axis;
          if (axis->IsInstance<DenseFixedAxisNode>()) {
            sp_index = GetDenseValue(sp_iter);
          } else if (const auto* sf_axis = axis.as<SparseFixedAxisNode>()) {
            if (iterated_axis.get() == sf_axis) {
              sp_index = sp_iter;
            } else {
              sp_index = lower_bound(sf_axis->indices->data, GetDenseValue(sp_iter), std::move(l),
                                     std::move(r));
            }
          } else if (const auto* sv_axis = axis.as<SparseVariableAxisNode>()) {
            sp_index = lower_bound(sv_axis->indices->data, GetDenseValue(sp_iter), std::move(l),
                                   std::move(r));
          } else {
            LOG(FATAL) << "Cannot reach here";
          }
          break;
        }
        default: {  // kind == SpIterKind::kSparseVariable
          CHECK(!axis->IsInstance<DenseVariableAxisNode>());
          CHECK(sp_iter->axis.defined());
          const Axis& iterated_axis = sp_iter->axis;
          if (const auto* df_axis = axis.as<DenseFixedAxisNode>()) {
            CHECK(ana_.CanProveEqual(sp_iter->max_extent, df_axis->length));
            sp_index = GetDenseValue(sp_iter);
          } else if (const auto* sf_axis = axis.as<SparseFixedAxisNode>()) {
            CHECK(ana_.CanProveEqual(sp_iter->max_extent, sf_axis->length));
            sp_index = lower_bound(sf_axis->indices->data, GetDenseValue(sp_iter), std::move(l),
                                   std::move(r));
          } else if (const auto* sv_axis = axis.as<SparseVariableAxisNode>()) {
            CHECK(ana_.CanProveEqual(sp_iter->max_extent, sv_axis->length));
            if (iterated_axis.get() == sv_axis) {
              sp_index = sp_iter;
            } else {
              sp_index = lower_bound(sv_axis->indices->data, GetDenseValue(sp_iter), std::move(l),
                                     std::move(r));
            }
          } else {
            LOG(FATAL) << "Cannot reach here";
          }
          break;
        }
      }

      // Stage 2. Accumulate the lowered index.
      lowered_index =
          PartialLowerIndex(std::move(lowered_index), sp_buffer->axes[i], std::move(sp_index));
    }

    return lowered_index;
  }

  /*!
   * \brief Compupte the partially lowered index.
   * \param prev_lowered_index The lowered index accumulated over all axis prior to current axis.
   * \param axis Current axis.
   * \param index The sparse index on current axis.
   * \return The lowered index.
   */
  PrimExpr PartialLowerIndex(PrimExpr prev_lowered_index, const Axis& axis, PrimExpr index) {
    if (axis->IsInstance<DenseFixedAxisNode>()) {
      return ana_.Simplify(std::move(prev_lowered_index) * axis->length + std::move(index));
    } else if (const auto* sf_axis = axis.as<SparseFixedAxisNode>()) {
      return ana_.Simplify(std::move(prev_lowered_index) * sf_axis->nnz_cols + std::move(index));
    } else if (const auto* dv_axis = axis.as<DenseVariableAxisNode>()) {
      return ana_.Simplify(
          add(BufferLoad(dv_axis->indptr, {std::move(prev_lowered_index)}), std::move(index)));
    } else if (const auto* sv_axis = axis.as<SparseVariableAxisNode>()) {
      return ana_.Simplify(
          add(BufferLoad(sv_axis->indptr, {std::move(prev_lowered_index)}), std::move(index)));
    }
    LOG(FATAL) << "Cannot reach here";
    throw;
  }

  /*!
   * \brief Convert sparse iteration positions to dense coordinates.
   * \param sp_iter The sparse iterator.
   */
  PrimExpr GetDenseValue(SpIterVar sp_iter) {
    SpIterKind kind = sp_iter->kind;
    CHECK(kind == SpIterKind::kSparseFixed || kind == SpIterKind::kSparseVariable);
    Axis iterated_axis = sp_iter->axis;

    SparseBuffer iterated_buffer{nullptr};
    Array<PrimExpr> iters{nullptr};

    collector_.GetIteratedBufferAndDependentIters(sp_iter, &iterated_buffer, &iters);
    iters.push_back(sp_iter);
    PrimExpr lowered_indices = LowerIndices(std::move(iterated_buffer), iters);

    if (kind == SpIterKind::kSparseFixed) {
      return BufferLoad(Downcast<SparseFixedAxis>(iterated_axis)->indices,
                        {std::move(lowered_indices)});
    } else {
      return BufferLoad(Downcast<SparseVariableAxis>(iterated_axis)->indices,
                        {std::move(lowered_indices)});
    }
  }

  /*!
   * \brief Convert sparse buffer load node to buffer load node.
   * \param load The sparse buffer load node in AST.
   */
  PrimExpr VisitExpr_(const SparseBufferLoadNode* load) final {
    buffer_read_.insert(load->buffer.get());
    PrimExpr lowered_indices = LowerIndices(load->buffer, load->indices);
    return BufferLoad(load->buffer->data, {std::move(lowered_indices)});
  }

  /*!
   * \brief Convert sparse buffer store node to buffer store node.
   * \param store The sparse buffer store node in AST.
   */
  Stmt VisitStmt_(const SparseBufferStoreNode* store) final {
    buffer_write_.insert(store->buffer.get());
    PrimExpr value = ExprMutator::VisitExpr(store->value);
    PrimExpr lowered_indices = LowerIndices(store->buffer, store->indices);
    return BufferStore(store->buffer->data, std::move(value), {std::move(lowered_indices)});
  }

  /*!
   * \brief Rewrite sparse block to ordinary block.
   * \param sp_block The sparse block to be rewritten.
   */
  Stmt VisitStmt_(const SparseBlockNode* sp_block) {
    int n_iter = static_cast<int>(sp_block->sp_iter_vars.size());
    buffer_read_.clear();
    buffer_write_.clear();

    // Step 1. Recursively mutate the `init` field and the block body.
    Optional<Stmt> init =
        sp_block->init.defined() ? VisitStmt(sp_block->init.value()) : Optional<Stmt>(NullOpt);
    Stmt body = VisitStmt(sp_block->body);

    // Step 2. Create the new loop vars.
    std::unordered_map<const VarNode*, PrimExpr> var_map;
    Array<Var> all_loop_vars;
    var_map.reserve(n_iter);
    for (const SpIterVar& sp_iter : sp_block->sp_iter_vars) {
      Var loop_var("v_" + sp_iter->var->name_hint);
      all_loop_vars.push_back(loop_var);
      var_map[sp_iter->var.get()] = loop_var;
    }

    // Step 3. Collet block iters and iter bindings.
    std::set<String> in_stack;
    in_stack.insert("root");
    /* A stack that stores block itervars in each block. */
    std::stack<Array<IterVar>> block_iters_st;
    /* A stack that stores itervar bindings in each block. */
    std::stack<Array<PrimExpr>> iter_bindings_st;
    /* A stack that stores generated loop vars in each block. */
    std::stack<Array<Var>> loop_vars_st;
    /* A stack that stores whether to place init block in each block. */
    std::stack<bool> place_init_st;
    /* An indicator that records whether init block has been set. */
    bool init_set = false;
    do {
      /* Block itervars of current block. */
      Array<IterVar> block_iters;
      /* Itervar bindings of current block. */
      Array<PrimExpr> iter_bindings;
      /* Axis names of current block. */
      Array<String> axis_names;
      /* Generated loop vars of current block. */
      Array<Var> loop_vars;
      /* An indicator that records whether there is reduction axis in current block. */
      bool has_reduction_var = false;
      for (int i = 0; i < n_iter; ++i) {
        SpIterVar sp_it_var = sp_block->sp_iter_vars[i];
        String axis_name = sp_it_var->axis->name;
        auto&& parent_axis = axis_tree_->parent.Get(axis_name);
        CHECK(parent_axis.defined()) << "Sparse IterVar not defined in Axis Tree.";
        String parent_axis_name = parent_axis.value();
        bool is_fixed_axis = sp_it_var->axis->is_fixed();
        /* Add itervar to current block when
         * - it's not used yet (not in stack) and
         *   - it's parent axis was used in outer blocks or
         *   - it's an iterator to a fixed axis.
         */
        if ((is_fixed_axis || in_stack.find(parent_axis_name) != in_stack.end()) &&
            in_stack.find(axis_name) == in_stack.end()) {
          loop_vars.push_back(all_loop_vars[i]);
          axis_names.push_back(std::move(axis_name));
          block_iters.push_back(SpIterVarToIterVar(sp_it_var, var_map));
          iter_bindings.push_back(all_loop_vars[i]);
          has_reduction_var |= sp_it_var->is_reduction;
        }
      }

      /* Tag axes in current block as "in-stack". */
      for (const String&& axis_name : axis_names) {
        in_stack.insert(std::move(axis_name));
      }

      /* Update stack. */
      if (!block_iters.empty()) {
        block_iters_st.push(std::move(block_iters));
        iter_bindings_st.push(std::move(iter_bindings));
        loop_vars_st.push(std::move(loop_vars));
        if (init_set) {
          place_init_st.push(false);
        } else {
          place_init_st.push(has_reduction_var);
          init_set |= has_reduction_var;
        }
      } else {
        break;
      }
    } while (true);

    // Step 4. Generate the read-region and write-retion of the block.
    Array<BufferRegion> reads{nullptr};
    Array<BufferRegion> writes{nullptr};
    GenerateReadWriteRegions(sp_block, &reads, &writes);

    // Step 5. Generate nested blocks and loops from innermost to outermost.
    int blk_counter = 0;
    while (!block_iters_st.empty()) {
      Array<IterVar> block_iters = std::move(block_iters_st.top());
      Array<PrimExpr> iter_bindings = std::move(iter_bindings_st.top());
      Array<Var> loop_vars = std::move(loop_vars_st.top());
      bool place_init = place_init_st.top();
      block_iters_st.pop();
      iter_bindings_st.pop();
      loop_vars_st.pop();
      place_init_st.pop();

      Map<String, ObjectRef> mapping;
      mapping.Set("sparse", Bool(true));
      String blk_name_hint = sp_block->name;
      if (blk_counter != 0) {
        blk_name_hint = blk_name_hint + "_" + std::to_string(blk_counter);
      }
      Block block(/*iter_vars=*/block_iters,
                  /*reads=*/reads,
                  /*writes=*/writes,
                  /*name_hint=*/blk_name_hint,
                  /*body=*/std::move(body),
                  /*init=*/place_init ? std::move(init) : NullOpt,
                  /*alloc_buffers=*/{},
                  /*match_buffers=*/{},
                  /*annotations=*/std::move(mapping),
                  /*span=*/sp_block->span);
      BlockRealize block_realize(std::move(iter_bindings), const_true(), std::move(block));
      // Generate outer loop and the block binding.
      Stmt loop = GenerateLoops(std::move(block_realize), block_iters, loop_vars);
      body = loop;
      blk_counter += 1;
    }

    return body;
  }

  /*!
   * \brief Convert sparse iterable variable to ordinary iterable variable.
   * \param sp_iter The sparse iterable variable to convert.
   * \param var_map The mapping from sparse iterable variable to corresponding ordinary iterable
   * variable.
   */
  IterVar SpIterVarToIterVar(const SpIterVar& sp_iter,
                             const std::unordered_map<const VarNode*, PrimExpr>& var_map) {
    PrimExpr extent{nullptr};

    SpIterKind kind = sp_iter->kind;
    if (kind == SpIterKind::kDenseFixed || kind == SpIterKind::kSparseFixed) {
      extent = sp_iter->max_extent;
    } else {
      SparseBuffer iterated_buffer{nullptr};
      Array<PrimExpr> dependent_iters{nullptr};
      collector_.GetIteratedBufferAndDependentIters(sp_iter, &iterated_buffer, &dependent_iters);
      PrimExpr lowered_indices = LowerIndices(std::move(iterated_buffer), dependent_iters);

      Buffer indptr{kind == SpIterKind::kDenseVariable
                        ? Downcast<DenseVariableAxis>(sp_iter->axis)->indptr
                        : Downcast<SparseVariableAxis>(sp_iter->axis)->indptr};
      PrimExpr l = BufferLoad(indptr, {lowered_indices});
      PrimExpr r = BufferLoad(indptr, {add(lowered_indices, 1)});
      extent = sub(r, l);
    }

    // Substitute the iteration vars in the expression with the loop vars.
    return IterVar(Range::FromMinExtent(0, Substitute(std::move(extent), var_map)), sp_iter->var,
                   sp_iter->is_reduction ? kCommReduce : kDataPar);
  }

  /*!
   * \brief generate read and write regions for sparse blocks.
   * \param sp_block the sparse blocks
   * \param reads pointer of array to read buffer regions.
   * \param writes pointer of array to write buffer regions.
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
   * \brief generated nested for-loops for sparse block.
   * \param block_iters The iterators defined in sparse blocks.
   * \param loop_vars The loop variables binded with block iterators.
   * \return The outermost loop.
   */
  Stmt GenerateLoops(Stmt body, const Array<IterVar>& block_iters, const Array<Var>& loop_vars) {
    int n_iter = static_cast<int>(block_iters.size());
    for (int i = n_iter - 1; i >= 0; --i) {
      const Range& dom = block_iters[i]->dom;
      body = For(loop_vars[i], dom->min, dom->extent, ForKind::kSerial, std::move(body));
    }
    return body;
  }

  AccessAndDependencyCollector collector_;
  AxisTree axis_tree_;
  arith::Analyzer ana_;
  std::unordered_set<const SparseBufferNode*> buffer_read_;
  std::unordered_set<const SparseBufferNode*> buffer_write_;
};

/*!
 * \brief Wrap the body statement with an empty root block.
 * \param body The body statements to wrap with.
 * \return The wrapped block.
 */
Stmt WrapWithRootBlock(Stmt body) {
  Block root_block({}, {}, {}, "root", std::move(body));
  body = BlockRealize({}, const_true(), std::move(root_block));
  return Stmt(body);
}

/*!
 * \brief Rewrite the given primitive function.
 * \param axis_tree The axis dependency tree.
 * \param f The Sparse-TIR primitive function to lower.
 * \return lowered primitive function in TIR.
 */
PrimFunc LowerSparseTIR(AxisTree axis_tree, PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    // Step 1. Update the PrimFunc's buffer map.
    fptr->buffer_map = UpdateBufferMap(f);
    // Step 2. Collect buffer access information and dependency.
    AccessAndDependencyCollector collector;
    collector.Collect(f->body);
    // Step 3. Lower indices.
    fptr->body = IndexTransformer(collector, axis_tree)(std::move(f->body));
    // Step 4. Wrap the function body with a root block.
    fptr->body = WrapWithRootBlock(std::move(fptr->body));
    return f;
  } else {
    return f;
  }
}

namespace transform {

/*!
 * \brief The lowering pass from TIR to Sparse TIR.
 * \param axis_tree The axis dependency tree.
 */
Pass LowerSparseTIR(AxisTree axis_tree) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerSparseTIR(std::move(axis_tree), std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerSparseTIR", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerSparseTIR").set_body_typed(LowerSparseTIR);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
