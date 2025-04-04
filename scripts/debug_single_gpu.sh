#!/bin/bash

# 单卡调试训练脚本
# 使用方法：bash scripts/debug_single_gpu.sh

# 使用单卡训练 时间序列预测任务
python run.py \
  --is_training 1 \
  --model_id debug_single \
  --model UniTS \
  --data ETTh1 \
  --features M \
  --batch_size 32 \
  --d_model 512 \
  --e_layers 2 \
  --train_epochs 3 \
  --task_data_config_path exp/all_task.yaml \
  --single_gpu \
  --num_workers 0 \
  --acc_it 1 \
  --patch_len 16 \
  --stride 8 

# 单卡预训练脚本
# python run_pretrain.py \
#   --is_training 1 \
#   --model_id debug_single_pretrain \
#   --model UniTS \
#   --data ETTh1 \
#   --features M \
#   --batch_size 32 \
#   --d_model 512 \
#   --e_layers 2 \
#   --train_epochs 3 \
#   --task_data_config_path exp/all_task_pretrain.yaml \
#   --single_gpu \
#   --num_workers 0 \
#   --acc_it 1 \
#   --patch_len 16 \
#   --stride 8 \
#   --min_mask_ratio 0.5 \
#   --max_mask_ratio 0.8 