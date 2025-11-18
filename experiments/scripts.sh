# Baseline pretraining scripts
python3 experiments/pretraining.py --rank=0 --seed=42 --path="datasets" --epoch=45 --dataset="omniglot" --lr=0.0001 --name="baseline-gpu" --save_interval=15 --eval_interval=5 --gpu=7
python3 experiments/pretraining.py --rank=0 --seed=42 --path="datasets" --epoch=45 --dataset="omniglot" --lr=0.0001 --weight_decay=0.01 --name="baseline-gpu-adamw" --save_interval=15 --eval_interval=5 --gpu=7


# Baseline finetuning scripts
python3 experiments/finetuning.py --rank=0 --seed=42 --path="datasets" --epoch=45 --dataset="omniglot" --lr=0.0001 --name="scratch-gpu" --save_interval=15 --eval_interval=5 --gpu=7 --model_path="checkpoints/pretrain_omniglot/baseline-gpu/checkpoint_45.pt"
python3 experiments/finetuning.py --rank=0 --seed=42 --path="datasets" --epoch=45 --dataset="omniglot" --lr=0.0001 --name="baseline-gpu-notall" --save_interval=15 --eval_interval=1 --gpu=7 --model_path="checkpoints/pretrain_omniglot/baseline-gpu-notall/checkpoint_45.pt"
python3 experiments/finetuning.py --rank=0 --seed=42 --path="datasets" --epoch=45 --dataset="omniglot" --lr=0.0001 --name="baseline-gpu-adamwanchored" --save_interval=15 --eval_interval=1 --gpu=7 --model_path="checkpoints/pretrain_omniglot/baseline-gpu-adamw/checkpoint_45.pt"
