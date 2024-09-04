#!/bin/zsh

# Change to the specified directory
cd /mnt/arc/yygx/pkgs_baselines/LIBERO/libero || { echo "Directory change failed"; exit 1; }

# Execute the sbatch command with the wrapped commands
sbatch --gpus 1 --cpus-per-task=8 \
       -o /mnt/arc/yygx/pkgs_baselines/LIBERO/libero/outs/debug_%j.out \
       -J eval \
       --wrap="
       export CUDA_VISIBLE_DEVICES=0 && \
       export MUJOCO_EGL_DEVICE_ID=0 && \
       python lifelong/train_skills.py is_split=false is_customize_task=false version=\"train_base_90_v0\" benchmark_name=libero_90 policy=bc_transformer_policy lifelong=base && \
       python lifelong/train_skills.py version=\"shelf_move_potato_v1_finetune\" data.seq_len=30 train.batch_size=32 train.optimizer.kwargs.lr=1e-4 && \
       python lifelong/train_skills.py version=\"shelf_move_potato_v1_finetune\" data.seq_len=30 train.batch_size=32 train.optimizer.kwargs.lr=1e-5 && \
       python lifelong/train_skills.py version=\"shelf_move_potato_v1_finetune\" data.seq_len=30 train.batch_size=64 train.optimizer.kwargs.lr=1e-4
       "
#export CUDA_VISIBLE_DEVICES=7 && export MUJOCO_EGL_DEVICE_ID=7 && python lifelong/train_skills.py is_split=false is_customize_task=true version="train_base_90_v0" benchmark_name=libero_90 policy=bc_transformer_policy lifelong=base

