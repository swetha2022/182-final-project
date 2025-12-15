#!/bin/bash
for PRETRAIN_RATIO in 0.0001 0.005 0.01 0.05;
do
    for ITERATION in $(seq 1 3);
    do
        python3 experiments/finetuning.py \
            --rank=0 \
            --seed=$ITERATION \
            --path="datasets" \
            --epoch=45 \
            --dataset="omniglot" \
            --lr=0.0001 \
            --name="baseline-adam-with-pretrain$PRETRAIN_RATIO" \
            --save_interval=15 \
            --eval_interval=5 \
            --gpu=4 \
            --model_path="checkpoints/pretrain_omniglot/baseline-adamw-full/checkpoint_45.pt" \
            --pretrain_ratio=$PRETRAIN_RATIO \
            --optimizer="adam"
    done
done