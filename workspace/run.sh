#!/bin/bash

tag=$1
device=$2


cd ..
echo "Processing: full/$tag"

cd "./dataset/full/$tag" || exit
CUDA_VISIBLE_DEVICES="$device" python train.py
CUDA_VISIBLE_DEVICES="$device" python finetune.py

cd "../../../workspace" || exit
bash launch.sh "$tag" "$device"
CUDA_VISIBLE_DEVICES="$device" python generate.py "$tag"
CUDA_VISIBLE_DEVICES="$device" python evaluate.py "$tag"
acd ..