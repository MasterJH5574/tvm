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

#include "../../support/utils.h"
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
 * \brief Compupte the partially lowered index.
 * \param prev_offset The lowered index accumulated over all axis prior to current axis.
 * \param axis Current axis.
 * \param index The sparse index on current axis.
 * \return The lowered index.
 */
PrimExpr AggregateOffset(PrimExpr prev_offset, const Axis& axis, PrimExpr index,
                         arith::Analyzer* ana_) {
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
  return ana_->Simplify(new_offset);
}

/*! \brief Storing the context information of a sparse block. */
class SparseBlockCtx {
 public:
  class Scope {
   public:
    // move constructor
    explicit Scope(Scope&& other)
        : sp_iter_var_map_(std::move(other.sp_iter_var_map_)),
          offset_(std::move(other.offset_)),
          parent_(std::move(parent_)),
          blk_name_(std::move(blk_name_)) {}

    // default constructor
    explicit Scope(String blk_name, Array<SpIterVar> sp_iter_vars, AxisTree tree)
        : blk_name_(std::move(blk_name)) {
      std::unordered_map<String, SpIterVar> axis_name_sp_iter_map_;
      // initialize sparse iter var dependency map.
      for (const SpIterVar& sp_iter_var : sp_iter_vars) {
        axis_name_sp_iter_map_[sp_iter_var->axis->name] = sp_iter_var;
        sp_iter_var_map_[sp_iter_var->var.get()] = sp_iter_var;
      }

      // collect parents.
      for (const SpIterVar& sp_iter_var : sp_iter_vars) {
        String axis_name = sp_iter_var->axis->name;
        const SpIterVarNode* node = sp_iter_var.get();
        if (support::EndsWith(axis_name, "_dense")) {
          // ends with "_dense", the axis is generated via to_dense
          parent_[node] = nullptr;
        } else {
          auto opt = tree->parent.Get(axis_name);
          CHECK(opt.defined()) << "Cannot find parent of axis " << axis_name << ".";
          String parent_axis_name = opt.value();
          if (parent_axis_name != "root") {
            auto it = axis_name_sp_iter_map_.find(parent_axis_name);
            CHECK(it != axis_name_sp_iter_map_.end())
                << "Cannot find sparse iter vars corresponding to parent axis " << parent_axis_name
                << " in current sparse block " << blk_name;
            parent_[node] = (it->second).get();
          } else {
            parent_[node] = nullptr;
          }
        }
      }

      // init offset_
      offset_[nullptr] = Integer(0);
    }

    Optional<SpIterVar> GetSparseIterVar(const VarNode* var_node) const {
      auto it = sp_iter_var_map_.find(var_node);
      if (it != sp_iter_var_map_.end()) {
        return it->second;
      } else {
        return NullOpt;
      }
    }

    /*!
     * \brief Get coordinate of corresding sparse iter var.
     * \param sp_iter_var The compressed iterator.
     */
    PrimExpr GetCoordinate(const SpIterVarNode* sp_iter_var) {
      const Axis& axis = sp_iter_var->axis;
      AxisKind kind = axis->kind();
      if (kind == AxisKind::kDenseFixed || kind == AxisKind::kDenseVariable) {
        // if dense, just return the value.
        return sp_iter_var->var;
      }

      PrimExpr offset = GetOffset(sp_iter_var);
      if (kind == AxisKind::kSparseFixed) {
        return BufferLoad(Downcast<SparseFixedAxis>(axis)->indices, {std::move(offset)});
      } else {  // AxisKind::kSparseVariable
        return BufferLoad(Downcast<SparseVariableAxis>(axis)->indices, {std::move(offset)});
      }
    }

    /*! \brief TODO
     */
    PrimExpr GetOffset(const SpIterVarNode* sp_iter_var) {
      auto it = offset_.find(sp_iter_var);
      if (it != offset_.end()) {
        return it->second;
      } else {
        PrimExpr prev_off = GetOffset(parent_[sp_iter_var]);
        PrimExpr new_off = AggregateOffset(prev_off, sp_iter_var->axis, sp_iter_var->var, &ana_);
        offset_[sp_iter_var] = new_off;
        return new_off;
      }
    }

    /*! \brief TODO
     */
    std::tuple<PrimExpr, PrimExpr> GetIndicesRange(const SpIterVarNode* sp_iter_var) {
      PrimExpr prev_off = GetOffset(parent_[sp_iter_var]);
      const Axis& axis = sp_iter_var->axis;
      return {AggregateOffset(prev_off, axis, Integer(0), &ana_),
              AggregateOffset(add(prev_off, 1), axis, Integer(0), &ana_)};
    }

    /*! \brief TODO
     */
    const String GetBlockName() const { return blk_name_; }

   private:
    std::unordered_map<const VarNode*, SpIterVar> sp_iter_var_map_;
    std::unordered_map<const SpIterVarNode*, PrimExpr> offset_;
    std::unordered_map<const SpIterVarNode*, const SpIterVarNode*> parent_;
    arith::Analyzer ana_;
    String blk_name_;
  };

  explicit SparseBlockCtx(AxisTree tree) : tree_(std::move(tree)) {}

  void EnterScope(const SparseBlockNode* sp_block) {
    stack_.emplace_back(sp_block->name, sp_block->sp_iter_vars, tree_);
  }

  void ExitScope() { stack_.pop_back(); }

  Optional<SpIterVar> GetSparseIterVar(const VarNode* var_node) const {
    return local()->GetSparseIterVar(var_node);
  }

  PrimExpr GetCoordinate(const SpIterVarNode* node) { return local()->GetCoordinate(node); }

  std::tuple<PrimExpr, PrimExpr> GetIndicesRange(const SpIterVarNode* sp_iter_var) {
    return local()->GetIndicesRange(sp_iter_var);
  }

  const String GetBlockName() const { return local()->GetBlockName(); }

 private:
  std::vector<Scope> stack_;
  AxisTree tree_;

  inline Scope* local() const { return const_cast<Scope*>(&stack_.back()); }
};

/*! \brief Storing the context information of a sparse buffer. */
class SparseBufferCtx {
 public:
  class Scope {
   public:
    // move constructor
    explicit Scope(Scope&& other)
        : buf_name_(std::move(other.buf_name_)),
          axes_(std::move(other.axes_)),
          offsets_(std::move(other.offsets_)),
          matches_(std::move(other.matches_)),
          sp_blk_ctx_(std::move(other.sp_blk_ctx_)) {}

    // default constructor
    explicit Scope(String buf_name, Array<Axis> axes, const SparseBlockCtx* sp_blk_ctx)
        : buf_name_(std::move(buf_name)), axes_(std::move(axes)), sp_blk_ctx_(sp_blk_ctx) {
      offsets_.emplace_back(Integer(0));
      matches_.emplace_back(true);
    }

    void Register(int idx, PrimExpr coordinate, PrimExpr orig_idx) {
      ICHECK(idx + 1 == int(offsets_.size()))
          << "Cannot register coordinate of index " << std::to_string(idx) << " at this time";
      const Axis& axis = GetAxis(idx);

      // update matches boolean array
      if (!matches_.back()) {
        // previous axies doesn't match.
        matches_.emplace_back(false);
      } else {
        const VarNode* node = orig_idx.as<VarNode>();
        auto it = sp_blk_ctx_->GetSparseIterVar(node);
        if (!it.defined()) {
          // current coordinate is not a single sparse iter var
          matches_.emplace_back(false);
        } else {
          const SpIterVar& sp_iter_var = it.value();
          // whether the axis current coordinate refers to matches the corresponding sparse buffer.
          matches_.emplace_back(axis->name == sp_iter_var->axis->name);
        }
      }

      // update offset
      PrimExpr new_offset = AggregateOffset(offsets_.back(), axis, std::move(coordinate), &ana_);
      offsets_.emplace_back(std::move(new_offset));
    }

    const Axis& GetAxis(int idx) const {
      auto && ret = axes_[idx];
      return ret;
    }

    const inline bool MatchWithSpBlock() const { return matches_.back(); }

    std::tuple<PrimExpr, PrimExpr> GetIndicesRange(int idx) {
      const Axis& axis = axes_[idx];
      return {AggregateOffset(offsets_[idx], axis, Integer(0), &ana_),
              AggregateOffset(add(offsets_[idx], 1), axis, Integer(0), &ana_)};
    }

   private:
    String buf_name_;
    Array<Axis> axes_;
    arith::Analyzer ana_;
    std::vector<PrimExpr> offsets_;
    std::vector<bool> matches_;
    const SparseBlockCtx* sp_blk_ctx_;
  };

  explicit SparseBufferCtx(AxisTree tree) : tree_(std::move(tree)) {}

  void EnterScope(const SparseBuffer& sp_buf, const SparseBlockCtx* sp_blk_ctx) {
    stack_.emplace_back(sp_buf->name, sp_buf->axes, sp_blk_ctx);
  }

  void ExitScope() { stack_.pop_back(); }

  const Axis& GetAxis(int idx) const { 
    auto&& ret = local()->GetAxis(idx);
    return ret;
  }

  const inline bool MatchWithSpBlock() const { return local()->MatchWithSpBlock(); }

  std::tuple<PrimExpr, PrimExpr> GetIndicesRange(int idx) { return local()->GetIndicesRange(idx); }

  void Register(int idx, PrimExpr coordinate, PrimExpr orig_idx) { local()->Register(idx, std::move(coordinate), std::move(orig_idx)); }

 private:
  AxisTree tree_;
  arith::Analyzer ana_;
  std::vector<Scope> stack_;

  inline Scope* local() const { return const_cast<Scope*>(&stack_.back()); }
};

/*!
 * \brief Rewrite indices in sparse buffers to indices in corresponding data
 * buffers.
 */
class IndexTransformer : public StmtExprMutator {
 public:
  explicit IndexTransformer(const AxisTree& axis_tree)
      : axis_tree_(axis_tree), sp_blk_ctx_(axis_tree), sp_buf_ctx_(axis_tree) {}

 private:
  // Sparse block context stack;
  SparseBlockCtx sp_blk_ctx_;
  // Sparse buffer context stack;
  SparseBufferCtx sp_buf_ctx_;

  PrimExpr ViewIndexInAxis(int idx, PrimExpr index) {
    // decompress index to coordinate on iterator axis.
    // the index might not be a single var node, use visitor to recursive construct the coordinate.
    PrimExpr coordinate = ExprMutator::VisitExpr(index);
    const Axis& axis = sp_buf_ctx_.GetAxis(idx);
    // register to sparse buffer scope
    sp_buf_ctx_.Register(idx, coordinate, index);

    PrimExpr offset = index;
    // compress coordinate to index on sparse buffer axis.
    if (!sp_buf_ctx_.MatchWithSpBlock()) {
      switch (axis->kind()) {
        case AxisKind::kDenseFixed:
        case AxisKind::kDenseVariable:
          offset = coordinate;
          break;
        case AxisKind::kSparseFixed: {
          auto sf_axis = axis.as<SparseFixedAxisNode>();
          PrimExpr l, r;
          std::tie(l, r) = sp_buf_ctx_.GetIndicesRange(idx);
          offset = lower_bound(sf_axis->indices->data, coordinate, l, r);
          break;
        }
        case AxisKind::kSparseVariable:
          auto sv_axis = axis.as<SparseVariableAxisNode>();
          PrimExpr l, r;
          std::tie(l, r) = sp_buf_ctx_.GetIndicesRange(idx);
          offset = lower_bound(sv_axis->indices->data, coordinate, l, r);
          break;
      }
    }

    return offset;
  }

  PrimExpr ComputeOffset(SparseBuffer sp_buffer, const Array<PrimExpr>& indices) {
    int num_lowered_indices = static_cast<int>(indices.size());
    ICHECK_LE(num_lowered_indices, sp_buffer->ndim());

    PrimExpr offset = Integer(0);
    for (int i = 0; i < num_lowered_indices; ++i) {
      const Axis& axis = sp_buffer->axes[i];
      const PrimExpr& index = indices[i];

      offset = AggregateOffset(offset, axis, ViewIndexInAxis(i, index), &ana_);
    }
    return offset;
  }

  /*!
   * \brief Change sparse iters to coordinates.
   * \param v The variable node.
   */
  PrimExpr VisitExpr_(const VarNode* v) final {
    auto it = sp_blk_ctx_.GetSparseIterVar(v);
    if (it.defined()) {
      return sp_blk_ctx_.GetCoordinate(it.value().get());
    } else{
      return GetRef<PrimExpr>(v);
    }
  }

  /*!
   * \brief Convert sparse buffer load node to buffer load node.
   * \param load The sparse buffer load node in AST.
   */
  PrimExpr VisitExpr_(const SparseBufferLoadNode* load) final {
    buffer_read_.insert(load->buffer.get());
    sp_buf_ctx_.EnterScope(load->buffer, &sp_blk_ctx_);
    PrimExpr lowered_indices = ComputeOffset(load->buffer, load->indices);
    sp_buf_ctx_.ExitScope();
    return BufferLoad(load->buffer->data, {std::move(lowered_indices)});
  }

  /*!
   * \brief Convert sparse buffer store node to buffer store node.
   * \param store The sparse buffer store node in AST.
   */
  Stmt VisitStmt_(const SparseBufferStoreNode* store) final {
    buffer_write_.insert(store->buffer.get());
    PrimExpr value = ExprMutator::VisitExpr(store->value);
    sp_buf_ctx_.EnterScope(store->buffer, &sp_blk_ctx_);
    PrimExpr lowered_indices = ComputeOffset(store->buffer, store->indices);
    sp_buf_ctx_.ExitScope();
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

    // Step 1. Enter new sparse block context.
    sp_blk_ctx_.EnterScope(sp_block);

    // Step 2. Recursively mutate the `init` field and the block body.
    Optional<Stmt> init =
        sp_block->init.defined() ? VisitStmt(sp_block->init.value()) : Optional<Stmt>(NullOpt);

    Stmt body = VisitStmt(sp_block->body);

    // Step 3. Create the new loop vars.
    std::unordered_map<const VarNode*, PrimExpr> var_map;
    Array<Var> all_loop_vars;
    var_map.reserve(n_iter);
    for (const SpIterVar& sp_iter_var : sp_block->sp_iter_vars) {
      Var loop_var("v_" + sp_iter_var->var->name_hint);
      all_loop_vars.push_back(loop_var);
      var_map[sp_iter_var->var.get()] = loop_var;
    }

    // Step 4. Collet block iters and iter bindings.
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

    // Step 5. Generate the read-region and write-retion of the block.
    Array<BufferRegion> reads{};
    Array<BufferRegion> writes{};
    GenerateReadWriteRegions(sp_block, &reads, &writes);

    // Step 6. Generate nested blocks and loops from innermost to outermost.
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

    // Step 7: Exit sparse block context.
    sp_blk_ctx_.ExitScope();

    return body;
  }

  /*!
   * \brief Convert sparse iterable variable to ordinary iterable variable.
   * \param sp_iter_var The sparse iterable variable to convert.
   * \param var_map The mapping from sparse iterable variable to corresponding ordinary iterable
   * variable.
   */
  IterVar SpIterVarToIterVar(const SpIterVar& sp_iter_var,
                             const std::unordered_map<const VarNode*, PrimExpr>& var_map) {
    PrimExpr extent{nullptr};
    AxisKind kind = sp_iter_var->axis->kind();
    switch (kind) {
      case AxisKind::kDenseFixed: {
        extent = sp_iter_var->max_extent;
        break;
      }
      case AxisKind::kSparseFixed: {
        const auto sp_fixed_axis = (sp_iter_var->axis).as<SparseFixedAxisNode>();
        extent = sp_fixed_axis->nnz_cols;
        break;
      }
      case AxisKind::kDenseVariable:
        // TODO(zihao): need discussion.
        break;
      case AxisKind::kSparseVariable: {
        PrimExpr l, r;
        std::tie(l, r) = sp_blk_ctx_.GetIndicesRange(sp_iter_var.get());
        extent = sub(r, l);
        break;
      }
    }

    // Substitute the iteration vars in the expression with the loop vars.
    return IterVar(Range::FromMinExtent(0, Substitute(std::move(extent), var_map)),
                   sp_iter_var->var, sp_iter_var->is_reduction ? kCommReduce : kDataPar);
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
    // Step 2. Lower indices.
    fptr->body = IndexTransformer(axis_tree)(std::move(f->body));
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
