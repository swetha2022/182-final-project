python utils/calculate_delta_theta_norm.py \
    --model_type alexnet \
    --pretrain_lr 0.0005 \
    --pretrain_weight_decay 0.00001 \
    --finetune1_pretrain_opt muon \
    --finetune1_finetune_opt adam \
    --finetune1_lr 0.00005 \
    --finetune1_ratio 0.05 \
    --finetune2_pretrain_opt muon \
    --finetune2_finetune_opt muon \
    --finetune2_lr 0.0001 \
    --finetune2_ratio 0.05

python utils/calculate_delta_theta_norm.py \
    --model_type alexnet \
    --pretrain_lr 0.00005 \
    --pretrain_weight_decay 0.0001 \
    --finetune1_pretrain_opt adam \
    --finetune1_finetune_opt adam \
    --finetune1_lr 0.00005 \
    --finetune1_ratio 0.05 \
    --finetune2_pretrain_opt adam \
    --finetune2_finetune_opt muon \
    --finetune2_lr 0.0001 \
    --finetune2_ratio 0.05

python utils/calculate_delta_theta_norm.py \
    --model_type alexnet \
    --pretrain_lr 0.0005 \
    --pretrain_weight_decay 0.00001 \
    --finetune1_pretrain_opt muon \
    --finetune1_finetune_opt adam \
    --finetune1_lr 0.00005 \
    --finetune1_ratio 0.01 \
    --finetune2_pretrain_opt muon \
    --finetune2_finetune_opt muon \
    --finetune2_lr 0.0001 \
    --finetune2_ratio 0.01

python utils/calculate_delta_theta_norm.py \
    --model_type alexnet \
    --pretrain_lr 0.00005 \
    --pretrain_weight_decay 0.0001 \
    --finetune1_pretrain_opt adam \
    --finetune1_finetune_opt adam \
    --finetune1_lr 0.00005 \
    --finetune1_ratio 0.01 \
    --finetune2_pretrain_opt adam \
    --finetune2_finetune_opt muon \
    --finetune2_lr 0.0001 \
    --finetune2_ratio 0.01