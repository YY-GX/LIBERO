#!/bin/zsh

# Change to the specified directory
cd /mnt/arc/yygx/pkgs_baselines/LIBERO/libero || { echo "Directory change failed"; exit 1; }

# Execute the sbatch command with the wrapped commands
sbatch --gpus 1 --cpus-per-task=8 \
       -o /mnt/arc/yygx/pkgs_baselines/LIBERO/libero/outs/debug_%j.out \
       -J eval \
       --wrap="
       export CUDA_VISIBLE_DEVICES=1 && \
       export MUJOCO_EGL_DEVICE_ID=1 && \
       python lifelong/evaluate_on_modified_task.py --seed 1 --benchmark \"modified_libero\" --policy \"bc_transformer_policy\" \
        --algo \"base\" --task_id 0 --load_task 0 --version \"modify_v0\" --device_id 0 --save-videos --seed 10000 \
        --model_path \"/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/experiments/libero_90/train_base_90_v0/Sequential/BCTransformerPolicy_seed10000/00_09/task0_model.pth\"
       "


