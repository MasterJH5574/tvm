set -euxo pipefail

RPC_HOST="127.0.0.1"
RPC_PORT="4446"
RPC_KEY="local"
# rpc_host = "172.16.2.241"
# rpc_port = 4445
# rpc_key = "amd-5900x"

NUM_TRIALS=64


run () {
    name=$1
    target=$2
    device=$3

    log_dir=$PWD/$name-$device
    mkdir -p $log_dir

    echo "Running model $name"
    python3 test_e2e_autotir.py             \
        --model "$name"                     \
        --target "$target"                  \
        --device "$device"                  \
        --num-trials $NUM_TRIALS            \
        --work-dir "$log_dir"               \
        --rpc-host "$RPC_HOST"              \
        --rpc-port "$RPC_PORT"              \
        --rpc-key "$RPC_KEY"                \
        2>&1 | tee "$log_dir/$name.log"
}


run resnet18 "llvm --num-cores=16" cpu