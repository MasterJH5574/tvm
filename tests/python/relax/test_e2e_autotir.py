# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm
import tvm.testing
import tvm.relay.testing
import tvm.meta_schedule as ms

from tvm import relax
from tvm import transform
from tvm.meta_schedule import tune_relax, EvolutionarySearchConfig

from tvm.script import relax as R
from tvm.relax.testing import relay_translator

import os
import logging
import argparse
import numpy as np


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model",
        type=str,
        required=True,
    )
    args.add_argument(
        "--target",
        type=str,
        required=True,
    )
    args.add_argument(
        "--device",
        type=str,
        required=True,
    )
    args.add_argument(
        "--num-trials",
        type=int,
        required=True,
    )
    args.add_argument(
        "--work-dir",
        type=str,
        required=True,
    )
    args.add_argument(
        "--rpc-host",
        type=str,
        required=True,
    )
    args.add_argument(
        "--rpc-port",
        type=int,
        required=True,
    )
    args.add_argument(
        "--rpc-key",
        type=str,
        required=True,
    )
    parsed = args.parse_args()
    parsed.target = tvm.target.Target(parsed.target)
    if parsed.target.attrs.get("mtriple", None) == "aarch64-linux-gnu":
        parsed.alloc_repeat = 3
    else:
        parsed.alloc_repeat = 1
    parsed.rpc_config = ms.runner.RPCConfig(
        tracker_host=parsed.rpc_host,
        tracker_port=parsed.rpc_port,
        tracker_key=parsed.rpc_key,
        session_timeout_sec=30,
    )
    parsed.rpc_workers = parsed.rpc_config.count_num_servers(allow_missing=False)
    parsed.device = tvm.cpu() if parsed.device == "cpu" else tvm.cuda()
    return parsed


logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)
ARGS = _parse_args()


def main():
    task_name = ARGS.model
    work_dir = ARGS.work_dir

    path_workload = os.path.join(work_dir, f"{task_name}_database_workload.json")
    path_tuning_record = os.path.join(work_dir, f"{task_name}_database_tuning_record.json")
    database = ms.database.JSONDatabase(
        path_workload=path_workload,
        path_tuning_record=path_tuning_record,
    )

    num_layers = 18
    batch_size = 1
    image_shape = (3, 224, 224)
    input_shape = (batch_size,) + image_shape

    relay_mod, params = tvm.relay.testing.resnet.get_workload(
        num_layers=num_layers, image_shape=image_shape, batch_size=batch_size, dtype="float32"
    )

    # translate the ResNet model from Relay to Relax
    relax_mod = relay_translator.from_relay(relay_mod["main"])
    assert isinstance(relax_mod, tvm.IRModule)

    # print(R.parser.astext(relax_mod))

    tune_relax(
        mod=relax_mod,
        target=ARGS.target,
        config=EvolutionarySearchConfig(
            num_trials_per_iter=64,
            num_trials_total=ARGS.num_trials,
        ),
        runner=ms.runner.RPCRunner(
            rpc_config=ARGS.rpc_config,
            alloc_repeat=3,
            max_workers=ARGS.rpc_workers,
        ),
        database=database,
        task_name=task_name,
        work_dir=work_dir,
        num_threads=os.cpu_count(),
    )

    with transform.PassContext(opt_level=0):
        ex_untuned, lib_untuned = relax.vm.build(relax_mod, ARGS.target)

    with transform.PassContext(opt_level=3):
        relax_mod_best = relax.transform.MetaScheduleApplyHistoryBest(database, ARGS.target)(
            relax_mod
        )
        # print(R.parser.astext(relax_mod_best))
        ex_tuned, lib_tuned = relax.vm.build(relax_mod_best, ARGS.target)

    vm_untuned = relax.vm.VirtualMachine(ex_untuned, ARGS.device, mod=lib_untuned)
    vm_tuned = relax.vm.VirtualMachine(ex_tuned, ARGS.device, mod=lib_tuned)

    data = tvm.nd.array(np.random.randn(*input_shape).astype("float32"), ARGS.device)

    def run_and_measure(vm: relax.vm.VirtualMachine, data, params):
        res = vm["main"](data, *list(params.values()))
        evaluator = vm.module.time_evaluator("main", ARGS.device, number=50)
        duration = evaluator(data, *list(params.values()))
        return res, duration

    res_untuned, time_untuned = run_and_measure(vm_untuned, data, params)
    res_tuned, time_tuned = run_and_measure(vm_tuned, data, params)

    tvm.testing.assert_allclose(res_tuned.numpy(), res_untuned.numpy(), rtol=1e-4, atol=1e-4)

    print(f"untuned resnet:\n{time_untuned}")
    print(f"  tuned resnet:\n{time_tuned}")


if __name__ == "__main__":
    main()
