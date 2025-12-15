for OPTIMIZER in adam sgd;
do 
    for ITERATION in $(seq 1 3);
    do
        python3 experiments/finetuning.py \
            --rank=0 \
            --seed=$ITERATION \
            --path="datasets" \
            --epoch=90 \
            --dataset="omniglot" \
            --lr=0.0001 \
            --name="baseline-$OPTIMIZER-probe" \
            --save_interval=15 \
            --eval_interval=5 \
            --gpu=4 \
            --model_path="checkpoints/pretrain_omniglot/baseline-adamw-full/checkpoint_45.pt" \
            --optimizer=$OPTIMIZER \
            --probe=True

        python3 experiments/finetuning.py \
            --rank=0 \
            --seed=$ITERATION \
            --path="datasets" \
            --epoch=90 \
            --dataset="omniglot" \
            --lr=0.0001 \
            --name="baseline-$OPTIMIZER-base-probe" \
            --save_interval=15 \
            --eval_interval=5 \
            --gpu=4 \
            --model_path="checkpoints/pretrain_omniglot/baseline-adamw-full/checkpoint_45.pt" \
            --optimizer=$OPTIMIZER \
            --base_probe=True
    done
done