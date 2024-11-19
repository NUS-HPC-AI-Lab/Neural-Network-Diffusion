#!/bin/bash

# set variable
gpu_ids="$2"
exec_file="$1"


# count gpus
IFS=',' read -r -a gpus <<< "$gpu_ids"
num_gpus=${#gpus[@]}

# find a usable port
find_open_port() {
  local port
  while true; do
    port=$(( (RANDOM % 32768) + 49152 ))
    if ! (echo >/dev/tcp/localhost/$port) &>/dev/null; then
      break
    fi
  done
  echo $port
}
main_process_port=$(find_open_port)
echo "Using main_process_port=$main_process_port"

# construct command
command="accelerate launch --main_process_port=$main_process_port --num_processes=$num_gpus --gpu_ids=$gpu_ids"
command+=" --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no"
if [ $num_gpus -ge 2 ]; then
  command+=" --multi_gpu"
fi
command+=" ./ablation/$exec_file.py"

# execute command
eval $command