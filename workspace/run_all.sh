#!/bin/bash

cls=$1
tag=$2
device=$3


cd ..
echo "Processing: $cls/$tag"

cd "./dataset/$cls/$tag" || exit
# CUDA_VISIBLE_DEVICES="$device" python train.py
# CUDA_VISIBLE_DEVICES="$device" python finetune.py

cd "../../../workspace" || exit
bash launch.sh "$cls" "$tag" "$device"
CUDA_VISIBLE_DEVICES="$device" python generate.py "$cls" "$tag"
CUDA_VISIBLE_DEVICES="$device" python evaluate.py "$cls" "$tag"
cd ..