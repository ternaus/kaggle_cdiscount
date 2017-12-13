#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0


for i in 0 1 2 3 5 6 7 8
do
    python predict.py --mode test --aug $i --device-ids 0 --model_type last --batch-size 1024 --model_name resnet50f_48
done