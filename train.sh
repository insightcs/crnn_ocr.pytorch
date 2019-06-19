#!/bin/bash

set -e
set -x

python train.py --arch=densenet \
                --data_root=$1 \
                --workers=0 \
                --image_w=650 \
                --image_h=32 \
                --checkpoint=./checkpoints \
                --alphabet=./chars/char_20868.txt \
                --optimizer=adadelta \
                --max_epoch=30 \
                --lr=0.01 \
                --batch_size=$2 \
                --decay_rate=0.1 \
                --weight_decay=0 \
                --drop_rate=0 \
                --display_interval=500 \
                --val_interval=1000 \
                --save_interval=1000 \
                --cuda

