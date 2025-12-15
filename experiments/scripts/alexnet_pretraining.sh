#!/bin/bash
for LR in 0.01 0.005 0.001; do
    for WEIGHT_DECAY in 0.0001 0.0005 0.00001; do
        for ITERATION in $(seq 1 4); do

            SEED=$((30 * ITERATION))

            python3 experiments/pretraining.py \
                --gpu=4 \
                --rank=0 \
                --seed=${SEED} \
                --path="datasets" \
                --epoch=90 \
                --dataset="omniglot" \
                --lr=${LR} \
                --weight_decay=${WEIGHT_DECAY} \
                --save_interval=15 \
                --eval_interval=5 \
                --name="alexnet-adam-pretrain-lr${LR}-wd${WEIGHT_DECAY}" \
                --checkpoint_folder="alexnet-adam-pretrain-lr${LR}-wd${WEIGHT_DECAY}-seed${SEED}" \
                --model_type="alexnet" \
                --optimizer="adam"

        done
    done
done
