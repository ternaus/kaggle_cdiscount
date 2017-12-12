#!/usr/bin/env bash

for i in 0 1 2 3

do
    python predict.py --mode test --aug $i --device-ids 0 --model_type last --batch-size 128
done