#!/usr/bin/env bash

for i in 4 5 6 7 8
do
    python predict.py --mode test --aug $i --device-ids 0 --model_type last --batch-size 128
done