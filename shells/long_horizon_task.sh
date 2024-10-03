#!/bin/zsh

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python libero/lifelong/eval_skill_chain_new.py --task_order_index 8 --seed 10000 --device_id 0 --benchmark "libero_90" --model_path_folder "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/experiments/libero_90/training_eval_skills_original_env/Sequential/BCRNNPolicy_seed10000/all" > out_long_2.out 2>&1 &

export CUDA_VISIBLE_DEVICES=1 && export MUJOCO_EGL_DEVICE_ID=1 && python libero/lifelong/eval_skill_chain_new.py --task_order_index 9 --seed 10000 --device_id 0 --benchmark "libero_90" --model_path_folder "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/experiments/libero_90/training_eval_skills_original_env/Sequential/BCRNNPolicy_seed10000/all" > out_long_3.out 2>&1 &

export CUDA_VISIBLE_DEVICES=2 && export MUJOCO_EGL_DEVICE_ID=2 && python libero/lifelong/eval_skill_chain_new.py --task_order_index 10 --seed 10000 --device_id 0 --benchmark "libero_90" --model_path_folder "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/experiments/libero_90/training_eval_skills_original_env/Sequential/BCRNNPolicy_seed10000/all" > out_long_4.out 2>&1 &

export CUDA_VISIBLE_DEVICES=3 && export MUJOCO_EGL_DEVICE_ID=3 && python libero/lifelong/eval_skill_chain_new.py --task_order_index 11 --seed 10000 --device_id 0 --benchmark "libero_90" --model_path_folder "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/experiments/libero_90/training_eval_skills_original_env/Sequential/BCRNNPolicy_seed10000/all" > out_long_5.out 2>&1 &

export CUDA_VISIBLE_DEVICES=4 && export MUJOCO_EGL_DEVICE_ID=4 && python libero/lifelong/eval_skill_chain_new.py --task_order_index 12 --seed 10000 --device_id 0 --benchmark "libero_90" --model_path_folder "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/experiments/libero_90/training_eval_skills_original_env/Sequential/BCRNNPolicy_seed10000/all" > out_long_6.out 2>&1 &

export CUDA_VISIBLE_DEVICES=5 && export MUJOCO_EGL_DEVICE_ID=5 && python libero/lifelong/eval_skill_chain_new.py --task_order_index 13 --seed 10000 --device_id 0 --benchmark "libero_90" --model_path_folder "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/experiments/libero_90/training_eval_skills_original_env/Sequential/BCRNNPolicy_seed10000/all" > out_long_7.out 2>&1 &

export CUDA_VISIBLE_DEVICES=6 && export MUJOCO_EGL_DEVICE_ID=6 && python libero/lifelong/eval_skill_chain_new.py --task_order_index 14 --seed 10000 --device_id 0 --benchmark "libero_90" --model_path_folder "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/experiments/libero_90/training_eval_skills_original_env/Sequential/BCRNNPolicy_seed10000/all" > out_long_8.out 2>&1 &

export CUDA_VISIBLE_DEVICES=7 && export MUJOCO_EGL_DEVICE_ID=7 && python libero/lifelong/eval_skill_chain_new.py --task_order_index 15 --seed 10000 --device_id 0 --benchmark "libero_90" --model_path_folder "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/experiments/libero_90/training_eval_skills_original_env/Sequential/BCRNNPolicy_seed10000/all" > out_long_9.out 2>&1 &

# Wait for all background processes to finish
wait
