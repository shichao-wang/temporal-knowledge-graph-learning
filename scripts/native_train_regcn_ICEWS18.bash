#! /usr/bin/bash 

python bin/native_train_regcn.py \
    --data-folder "./data/legacy/ICEWS18" \
    --hist-len 6 \
    --hidden-size 200 \
    --num-layers 2 \
    --kernel-size 3 \
    --channels 50 \
    --dropout 0.2 \
    --lr 1e-3
