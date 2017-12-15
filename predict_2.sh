#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0



python predict.py --mode test --aug 8 --device-ids 0 --model_type best --batch-size 1024 --model_name resnet101f_34
