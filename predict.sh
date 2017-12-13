#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

for i in 7 8

do
    python predict.py --mode test --aug $i --device-ids 0 --model_type last --batch-size 1024 --model_name resnet101f_34
done