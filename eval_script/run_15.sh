#!/bin/bash

set -euxo pipefail

TIME_TAG=`date +%y%m%d-%H%M%S`

export GPU_NUM_DEVICES=1
export PJRT_DEVICE=GPU

LOG_DIR=logs/e2e
mkdir -p $LOG_DIR

for bs in 1 16
do
    for model in align bert deberta densenet monodepth quantized resnet tridentnet
    for model in resnet
    do
        for compile in eager dynamo sys script sys-torchscript sys-tvm
        do
            rm -rf $LOG_DIR/$model.$bs.$compile.log
            LD_PRELOAD=build/ldlong.v3.9.12.so python3 run.py --bs $bs --model $model --compile $compile 2>&1 | tee $LOG_DIR/$model.$bs.$compile.log
        done
    done
done

wait
