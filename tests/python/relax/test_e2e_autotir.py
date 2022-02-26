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
from tvm.target.target import Target
from tvm.meta_schedule import tune_relax, EvolutionarySearchConfig

from tvm.relax.testing import relay_translator

import os
import time
import numpy as np


num_total_trials = 2000
# rpc_host = "172.16.2.241"
# rpc_port = 4445
# rpc_key = "amd-5900x"
rpc_host = "127.0.0.1"
rpc_port = 4446
rpc_key = "local"

rpc_config = ms.runner.RPCConfig(
    tracker_host=rpc_host,
    tracker_port=rpc_port,
    tracker_key=rpc_key,
    session_timeout_sec=30,
)
rpc_workers = rpc_config.count_num_servers(allow_missing=False)


def test_resnet_cpu():
    device = tvm.cpu(0)
    target = Target("llvm --num-cores=16")
    task_name = "resnet18-cpu-128"
    work_dir = "/home/rhlai/tvm/tests/python/relax/resnet18-cpu"
    rpc_runner = ms.runner.RPCRunner(
        rpc_config=rpc_config,
        alloc_repeat=3,
        max_workers=rpc_workers,
    )

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
    relay_mod, _ = tvm.relay.testing.resnet.get_workload(
        num_layers=num_layers, image_shape=image_shape, batch_size=batch_size, dtype="float32"
    )

    # translate the ResNet model from Relay to Relax
    relax_mod = relay_translator.from_relay(relay_mod["main"])
    assert isinstance(relax_mod, tvm.IRModule)

    # print(R.parser.astext(relax_mod))

    tune_relax(
        mod=relax_mod,
        target=target,
        config=EvolutionarySearchConfig(
            num_trials_per_iter=64,
            num_trials_total=64,
        ),
        runner=rpc_runner,
        task_name=task_name,
        work_dir=work_dir,
        num_threads=os.cpu_count(),
    )

    with transform.PassContext(opt_level=0):
        ex_untuned, lib_untuned = relax.vm.build(relax_mod, target)

    with transform.PassContext(opt_level=3):
        relax_mod_best = relax.transform.MetaScheduleApplyHistoryBest(database, target)(relax_mod)
        ex_tuned, lib_tuned = relax.vm.build(relax_mod_best, target)

    vm_untuned = relax.VirtualMachine(ex_untuned, device, mod=lib_untuned)
    vm_tuned = relax.VirtualMachine(ex_tuned, device, mod=lib_tuned)

    data = tvm.nd.array(np.random.randn(*input_shape).astype("float32"), device)

    def run_and_measure(vm, data):
        time_begin = time.time()
        print(type(vm["main"]))
        res = vm["main"](data)
        time_end = time.time()
        duration = time_end - time_begin
        return res, duration

    res_untuned, time_untuned = run_and_measure(vm_untuned, data)
    res_tuned, time_tuned = run_and_measure(vm_tuned, data)

    tvm.testing.assert_allclose(res_tuned.numpy(), res_untuned.numpy(), rtol=1e-4, atol=1e-4)

    print(f"untuned resnet: {time_untuned}")
    print(f"  tuned resnet: {time_tuned}")


if __name__ == "__main__":
    test_resnet_cpu()
