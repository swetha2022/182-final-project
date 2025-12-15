# Baseline pretraining scripts
python3 experiments/pretraining.py \
    --rank=0 \
    --seed=42 \
    --path="datasets" \
    --epoch=45 \
    --dataset="omniglot" \
    --lr=0.0001 \
    --weight_decay=0.01 \
    --name="baseline-adamw-full" \
    --save_interval=15 \
    --eval_interval=5 \
    --gpu=4 \
    --optimizer="adamw" \
    --num_tests=100 
python3 experiments/pretraining.py --rank=0 --seed=42 --path="datasets" --epoch=45 --dataset="omniglot" --lr=0.0001 --weight_decay=0.0 --name="baseline-adam" --save_interval=15 --eval_interval=5 --gpu=4 --optimizer="adamw" --num_tests=100 
python3 experiments/pretraining.py --rank=0 --seed=42 --path="datasets" --epoch=45 --dataset="omniglot" --lr=0.0001 --weight_decay=0.01 --name="baseline-sgd-momentum" --save_interval=15 --eval_interval=5 --gpu=4 --optimizer="sgd" --num_tests=100 --momentum=0.9


# Baseline finetuning scripts
python3 experiments/finetuning.py \
    --rank=0 \
    --seed=42 \
    --path="datasets" \
    --epoch=45 \
    --dataset="omniglot" \
    --lr=0.0001 \
    --name="baseline-adam-with-pretrain" \
    --save_interval=15 \
    --eval_interval=5 \
    --gpu=4 \
    --model_path="checkpoints/pretrain_omniglot/baseline-adamw-full/checkpoint_45.pt" \
    --pretrain_ratio=0.0001 \
    --optimizer="adamw"

python3 experiments/finetuning.py \
    --rank=0 \
    --seed=42 \
    --path="datasets" \
    --epoch=90 \
    --dataset="omniglot" \
    --lr=0.0001 \
    --name="baseline-adam-naive" \
    --save_interval=15 \
    --eval_interval=5 \
    --gpu=1 \
    --model_path="checkpoints/pretrain_omniglot/baseline-adamw-full/checkpoint_45.pt" \
    --optimizer="adamw"

python3 experiments/finetuning.py \
    --rank=0 \
    --seed=42 \
    --path="datasets" \
    --epoch=90 \
    --dataset="omniglot" \
    --lr=0.0001 \
    --name="baseline-adam-probe" \
    --save_interval=15 \
    --eval_interval=5 \
    --gpu=4 \
    --model_path="checkpoints/pretrain_omniglot/baseline-adamw-full/checkpoint_45.pt" \
    --optimizer="adamw" \
    --probe=True

python3 experiments/finetuning.py \
    --rank=0 \
    --seed=42 \
    --path="datasets" \
    --epoch=90 \
    --dataset="omniglot" \
    --lr=0.0001 \
    --name="baseline-sgd-probe" \
    --save_interval=15 \
    --eval_interval=5 \
    --gpu=4 \
    --model_path="checkpoints/pretrain_omniglot/baseline-adamw-full/checkpoint_45.pt" \
    --optimizer="sgd" \
    --probe=True

python3 experiments/finetuning.py \
    --rank=0 \
    --seed=42 \
    --path="datasets" \
    --epoch=90 \
    --dataset="omniglot" \
    --lr=0.0001 \
    --name="baseline-adam-base-probe" \
    --save_interval=15 \
    --eval_interval=5 \
    --gpu=1 \
    --model_path="checkpoints/pretrain_omniglot/baseline-adamw-full/checkpoint_45.pt" \
    --optimizer="adamw" \
    --base_probe=True

python3 experiments/finetuning.py \
    --rank=0 \
    --seed=42 \
    --path="datasets" \
    --epoch=90 \
    --dataset="omniglot" \
    --lr=0.0001 \
    --name="baseline-sgd-base-probe" \
    --save_interval=15 \
    --eval_interval=5 \
    --gpu=1 \
    --model_path="checkpoints/pretrain_omniglot/baseline-adamw-full/checkpoint_45.pt" \
    --optimizer="sgd" \
    --base_probe=True

python3 experiments/finetuning.py --rank=0 --seed=42 --path="datasets" --epoch=45 --dataset="omniglot" --lr=0.0001 --name="baseline-adamwanchored" --save_interval=15 --eval_interval=5 --gpu=1 --model_path="checkpoints/pretrain_omniglot/baseline-adamw/checkpoint_45.pt"
python3 experiments/finetuning.py --rank=0 --seed=42 --path="datasets" --epoch=45 --dataset="omniglot" --lr=0.0001 --name="baseline-adamwanchored-with-pretrain" --save_interval=15 --eval_interval=5 --gpu=4 --pretrain_ratio=0.1 --model_path="checkpoints/pretrain_omniglot/baseline-adamw/checkpoint_45.pt"

python3 experiments/finetuning.py --rank=0 --seed=42 --path="datasets" --epoch=45 --dataset="omniglot" --lr=0.0001 --name="scratch-adamw" --save_interval=15 --eval_interval=5 --gpu=1 --optimizer="adamw"
python3 experiments/finetuning.py --rank=0 --seed=42 --path="datasets" --epoch=45 --dataset="omniglot" --lr=0.0001 --name="scratch-adam" --save_interval=15 --eval_interval=5 --gpu=1 --optimizer="adamw" --weight_decay=0.0
python3 experiments/finetuning.py --rank=0 --seed=42 --path="datasets" --epoch=45 --dataset="omniglot" --lr=0.0001 --name="scratch-sgd" --save_interval=15 --eval_interval=5 --gpu=1 --optimizer="sgd" --momentum=0.0
python3 experiments/finetuning.py --rank=0 --seed=42 --path="datasets" --epoch=45 --dataset="omniglot" --lr=0.0001 --name="scratch-sgd-momentum" --save_interval=15 --eval_interval=5 --gpu=1 --optimizer="sgd" --momentum=0.9