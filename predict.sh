#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

for i in 6 7 8

do
    python predict.py --mode test --aug $i --device-ids 0 --model_type best --batch-size 1024 --model_name resnet50f_48
done