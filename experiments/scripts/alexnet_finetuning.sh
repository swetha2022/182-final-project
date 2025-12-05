#!/bin/bash
for LR in 0.005 0.001 0.0005; do
    for PRE_RATIO in 0.0 0.005 0.01 0.05; do
        for ITERATION in 3; do

            SEED=$((30 * ITERATION))

            python3 experiments/finetuning.py \
                --gpu=4 \
                --rank=0 \
                --seed=${SEED} \
                --path="datasets" \
                --epoch=90 \
                --dataset="omniglot" \
                --lr=${LR} \
                --weight_decay=0.0 \
                --pretrain_ratio=${PRE_RATIO} \
                --save_interval=15 \
                --eval_interval=5 \
                --name="alexnet-pretrain-muon-finetune-muon-lr${LR}-preratio${PRE_RATIO}" \
                --model_type="alexnet" \
                --model_path="checkpoints/pretrain_omniglot/alexnet-muon-pretrain-lr0.0005-wd0.00001-seed30/checkpoint_90.pt" \
                --checkpoint_folder="alexnet-pretrain-muon-finetune-muon-lr${LR}-preratio${PRE_RATIO}-seed${SEED}" \
                --optimizer="muon"

        done
    done
done