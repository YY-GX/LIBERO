#!/bin/zsh

# Change to the specified directory
cd /mnt/arc/yygx/pkgs_baselines/LIBERO/libero || { echo "Directory change failed"; exit 1; }

# Start building the sbatch command
sbatch_command="sbatch --gpus 1 --cpus-per-task=8 \
       -o /mnt/arc/yygx/pkgs_baselines/LIBERO/libero/outs/debug_%j.out \
       -J eval \
       --wrap=\"
       export CUDA_VISIBLE_DEVICES=0 && \
       export MUJOCO_EGL_DEVICE_ID=0 && \
"

# Loop over task_id and load_task from 0 to 9
for i in {0..9}; do
    sbatch_command+="python lifelong/evaluate.py --seed 1 --benchmark \\\"libero_90\\\" --policy \\\"bc_transformer_policy\\\" --algo \\\"base\\\" --run_id 2  --task_id $i --load_task $i --version \\\"train_base_90_v0\\\" --device_id 0 --save-videos --seed 10000 && \
"
done

# Close the wrap command
sbatch_command+="\""

# Execute the sbatch command
eval $sbatch_command
