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

#ifndef TVM_META_SCHEDULE_SCHEDULE_RULE_H_
#define TVM_META_SCHEDULE_SCHEDULE_RULE_H_

#include <tvm/tir/schedule/schedule.h>

namespace tvm {
namespace meta_schedule {

class TuneContext;

/*! \brief Rules to modify a block in a schedule. */
class ScheduleRuleNode : public runtime::Object {
 public:
  /*! \brief Virtual destructor. */
  virtual ~ScheduleRuleNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Initialize the design space generator with tuning context.
   * \param context The tuning context for initialization.
   * \note This method is supposed to be called only once before every other method.
   */
  virtual void InitializeWithTuneContext(const TuneContext& context) = 0;

  /*!
   * \brief Apply a schedule rule to the specific block in the given schedule.
   * \param sch The schedule to be modified.
   * \param block The specific block to apply the schedule rule.
   * \return The list of schedules generated by applying the schedule rule.
   */
  virtual runtime::Array<tir::Schedule> Apply(const tir::Schedule& sch,
                                              const tir::BlockRV& block) = 0;

  static constexpr const char* _type_key = "meta_schedule.ScheduleRule";
  TVM_DECLARE_BASE_OBJECT_INFO(ScheduleRuleNode, Object);
};

/*! \brief The schedule rule with customized methods on the python-side. */
class PyScheduleRuleNode : public ScheduleRuleNode {
 public:
  /*!
   * \brief The function type of `InitializeWithTuneContext` method.
   * \param context The tuning context for initialization.
   */
  using FInitializeWithTuneContext = runtime::TypedPackedFunc<void(const TuneContext&)>;
  /*!
   * \brief The function type of `Apply` method.
   * \param sch The schedule to be modified.
   * \param block The specific block to apply the schedule rule.
   * \return The list of schedules generated by applying the schedule rule.
   */
  using FApply =
      runtime::TypedPackedFunc<Array<tir::Schedule>(const tir::Schedule&, const tir::BlockRV&)>;
  /*!
   * \brief Get the schedule rule as string with name.
   * \return The string of the schedule rule.
   */
  using FAsString = runtime::TypedPackedFunc<String()>;

  /*! \brief The packed function to the `InitializeWithTuneContext` function. */
  FInitializeWithTuneContext f_initialize_with_tune_context;
  /*! \brief The packed function to the `Apply` function. */
  FApply f_apply;
  /*! \brief The packed function to the `AsString` function. */
  FAsString f_as_string;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `f_initialize_with_tune_context` is not visited
    // `f_apply` is not visited
    // `f_as_string` is not visited
  }

  void InitializeWithTuneContext(const TuneContext& context) final {
    ICHECK(f_initialize_with_tune_context != nullptr)
        << "PyScheduleRule's InitializeWithTuneContext method not implemented!";
    this->f_initialize_with_tune_context(context);
  }

  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block) final {
    ICHECK(f_apply != nullptr) << "PyScheduleRule's Apply method not implemented!";
    return this->f_apply(sch, block);
  }

  static constexpr const char* _type_key = "meta_schedule.PyScheduleRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyScheduleRuleNode, ScheduleRuleNode);
};

/*!
 * \brief Managed reference to ScheduleRuleNode
 * \sa ScheduleRuleNode
 */
class ScheduleRule : public runtime::ObjectRef {
 public:
  /*!
   * \brief Create an auto-inline rule that inlines spatial blocks if it satisfies some conditions
   * \param into_producer If allows to inline a block into its producer
   * \param into_consumer If allows to inline a block into its consumer
   * \param into_cache_only If it only allows to inline into a block generated by cache_read/write
   * \param inline_const_tensor Always inline constant tensors
   * \param disallow_if_then_else Always disallow if-then-else-like constructs
   * \param require_ordered Always require the read-to-write mapping to be ordered
   * \param require_injective Always require the read-to-write mapping to be injective
   * \param disallow_op The operators that are disallowed in auto inline
   * \return The schedule rule created
   */
  TVM_DLL static ScheduleRule AutoInline(bool into_producer,          //
                                         bool into_consumer,          //
                                         bool into_cache_only,        //
                                         bool inline_const_tensor,    //
                                         bool disallow_if_then_else,  //
                                         bool require_injective,      //
                                         bool require_ordered,        //
                                         Optional<Array<String>> disallow_op);
  /*!
   * \brief Create a mega rule: multi-level tiling with data reuse
   * \param structure The tiling structure. Recommended:
   * - 'SSRSRS' on CPU
   * - 'SSSRRSRS' on GPU
   * \param tile_binds For each level of tiles, which thread axis it is bound to. Recommended:
   * - NullOpt on CPU
   * - [blockIdx.x, vthread.x, threadIdx.x] on GPU
   * \param use_tensor_core Whether to apply tensor core wmma intrinsic for the computation
   * \param max_innermost_factor The maximum size of the innermost factor. NullOpt means no limit
   * \param vector_load_max_len The length of vector lane in vectorized cooperative fetching.
   * NullOpt means disable vectorization
   * \param reuse_read Data reuse configuration for reading. NullOpt means no reuse.
   * \param reuse_write Data reuse configuration for writing. NullOpt means no reuse.
   * \return The schedule rule created
   */
  TVM_DLL static ScheduleRule MultiLevelTiling(String structure,                             //
                                               Optional<Array<String>> tile_binds,           //
                                               bool use_tensor_core,                         //
                                               Optional<Integer> max_innermost_factor,       //
                                               Optional<Integer> vector_load_max_len,        //
                                               Optional<Map<String, ObjectRef>> reuse_read,  //
                                               Optional<Map<String, ObjectRef>> reuse_write);
  /*!
   * \brief Create a rule: add-rfactor to some blocks if needed
   * \param max_jobs_per_core The maximum number of jobs to be launched per CPU core. It sets the
   * uplimit of CPU parallelism, i.e. `num_cores * max_jobs_per_core`. Use -1 to disable
   * parallelism.
   * \param max_innermost_factor The maximum size of the innermost factor. NullOpt means no limit
   * \return The schedule rule created
   */
  TVM_DLL static ScheduleRule AddRFactor(int max_jobs_per_core,  //
                                         Optional<Integer> max_innermost_factor);
  /*!
   * \brief Create a schedule rule which applies cross-thread reduction to some reduction blocks
   * correspondingly when needed
   * \param thread_extents Candidates of thread axis extent (values are required to be positive).
   * \return The schedule rule created
   */
  TVM_DLL static ScheduleRule CrossThreadReduction(Array<Integer> thread_extents);
  /*!
   * \brief A rule that randomly select a compute-at location for a free block
   * \return The rule created
   */
  TVM_DLL static ScheduleRule RandomComputeLocation();
  /*!
   * \brief Mark parallelize, vectorize and unroll to each block correspondingly
   * \param max_jobs_per_core The maximum number of jobs to be launched per CPU core. It sets the
   * uplimit of CPU parallelism, i.e. `num_cores * max_jobs_per_core`. Use -1 to disable
   * parallelism.
   * \param max_vectorize_extent The maximum extent to be vectorized.
   * It sets the uplimit of the CPU vectorization. Use -1 to disable vectorization.
   * \param unroll_max_steps The maximum number of unroll steps to be done.
   * Use an empty array to disable unroll.
   * \param unroll_explicit Whether to explicitly unroll the loop, or just add a unroll pragma.
   * \return The schedule rule created
   */
  TVM_DLL static ScheduleRule ParallelizeVectorizeUnroll(int max_jobs_per_core,            //
                                                         int max_vectorize_extent,         //
                                                         Array<Integer> unroll_max_steps,  //
                                                         bool unroll_explicit);
  /*!
   * \brief Create a schedule rule with customized methods on the python-side.
   * \param f_initialize_with_tune_context The packed function of `InitializeWithTuneContext`.
   * \param f_apply The packed function of `Apply`.
   * \param f_as_string The packed function of `AsString`.
   * \return The schedule rule created.
   */
  TVM_DLL static ScheduleRule PyScheduleRule(
      PyScheduleRuleNode::FInitializeWithTuneContext f_initialize_with_tune_context,  //
      PyScheduleRuleNode::FApply f_apply,                                             //
      PyScheduleRuleNode::FAsString f_as_string);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleRule, ObjectRef, ScheduleRuleNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_SCHEDULE_RULE_H_
