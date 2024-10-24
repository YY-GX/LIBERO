import argparse
import sys
import os

# TODO:
#  1. pretrained model
#  2. initial states set
#  3. change to for loop to iterate tasks


os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark, task_orders
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, SequentialEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.metric import (
    raw_obs_to_tensor_obs,
)
from libero.lifelong.utils import (
    safe_device,
    torch_load_model,
)

from libero.lifelong.main import get_task_embs
import robomimic.utils.obs_utils as ObsUtils
from libero.lifelong.algos import get_algo_class
import warnings
import pickle
import wandb
import time
import copy

warnings.filterwarnings("ignore", category=DeprecationWarning)

algo_map = {
    "base": "Sequential",
    "er": "ER",
    "ewc": "EWC",
    "packnet": "PackNet",
    "multitask": "Multitask",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--model_path_folder", type=str,
                        default="/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/experiments/libero_90/training_eval_skills_original_env/Sequential/BCRNNPolicy_seed10000/all/")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["libero_10", "libero_90", "libero_spatial", "libero_object", "libero_goal", "yy_try",
                 "modified_libero"],
        default="libero_90"
    )
    parser.add_argument("--task_order_index", type=int, default=5)
    parser.add_argument("--seed", type=int, required=True, default=10000)
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()
    args.device_id = "cuda:" + str(args.device_id)
    return args

def initialize_robot_state(crr_state, robot_init_sim_state):
    # yy: 0: timestep; 1-40: states; 41-76: vel_info;
    modified_state = crr_state.copy()
    # initial robot states
    modified_state[1:10] = robot_init_sim_state[1:10]
    # zeroize all velocity related states
    modified_state[41:] = robot_init_sim_state[41:]
    return modified_state


def reset_env_init_states(env, obs, info, init_states_ls, env_num, task_indexes):
    obs_ls = []
    for k in range(env_num):
        if info[k]['is_init']:
            # yy: next task's initial state is extracted,
            #  and then passed to be modifed as I only wanna change robot related state
            init_state_ = initialize_robot_state(env.get_sim_state()[k], init_states_ls[task_indexes[k]][k, :])[
                None, ...]
            obs_ = env.set_init_state(init_state_, k)
            obs_ls.append(obs_[0])
        else:
            obs_ = obs[k]
            obs_ls.append(obs_)
    obs = np.stack(obs_ls)
    return obs


def main():
    args = parse_args()
    """
    Preparation for Evaluation
    """
    # Get the benchmarks
    benchmark = get_benchmark(args.benchmark)(args.task_order_index)
    n_tasks = benchmark.n_tasks
    task_id_ls = task_orders[args.task_order_index]
    task_idx_ls = [i for i in range(len(task_id_ls))]

    # Obtain language descriptions
    descriptions = [benchmark.get_task(i).language for i in range(n_tasks)]
    print("======= Tasks Language =======")
    print(f"{descriptions}")
    print("======= Tasks Language =======")

    save_dir = os.path.join(args.model_path_folder, f"long_horizon_task_id{args.task_order_index}_seed{args.seed}")
    print(f">> Create folder {save_dir}")
    os.system(f"mkdir -p {save_dir}")

    # yy: For collecting necessary list of items
    # For sequential env, need to obtain: cfg_ls, algo_ls, initial_states_ls
    cfg_ls, algo_ls, init_states_ls, task_ls = [], [], [], []
    task_embs = []
    for task_idx, task_id in enumerate(task_id_ls):  # task_id is the actual id of the task. task_idx is just the index.
        print(f">> Evaluate on original Task {task_id}")
        # Obtain useful info from saved model - checkpoints / cfg
        model_index = task_id
        model_path = args.model_path_folder
        model_path = os.path.join(model_path, f"task{model_index}_model.pth")
        if not os.path.exists(model_path):
            print(f">> {model_path} does NOT exist!")
            print(f">> Env_{task_id} evaluation fails.")
            continue
        sd, cfg, previous_mask = torch_load_model(
            model_path, map_location=args.device_id
        )

        # Modify some attributes of cfg via args
        cfg.benchmark_name = args.benchmark
        cfg.folder = get_libero_path("datasets")
        cfg.bddl_folder = get_libero_path("bddl_files")
        cfg.init_states_folder = get_libero_path("init_states")
        cfg.device = args.device_id
        # yy: cfg_ls here
        cfg_ls.append(cfg)

        # Create algo
        algo = safe_device(get_algo_class(algo_map["base"])(n_tasks, cfg), cfg.device)
        algo.policy.load_state_dict(sd)
        algo.eval()
        # yy: algo_ls here
        algo_ls.append(algo)

        # Obtain language embs & task
        task_embs += get_task_embs(cfg, descriptions)
        benchmark.set_task_embs(task_embs)
        task = benchmark.get_task(task_idx)
        # yy: task_ls here
        task_ls.append(task)

        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        indices = np.arange(cfg['eval']['n_eval']) % init_states.shape[0]
        # yy: init_states_ls here
        init_states_ls.append(init_states[indices])



    """
    Start Evaluation
    """
    cfg = cfg_ls[0]
    eval_task_id = []
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})

    save_stats_pth = os.path.join(
        save_dir,
        f"load_ori_{model_index}_on_ori_{task_id}.stats",
    )

    video_folder = os.path.join(
        save_dir,
        f"load_ori_{model_index}_on_ori_{task_id}_videos",
    )

    os.system(f"mkdir -p {video_folder}")

    with Timer() as t:
        # yy: video recorder preparation
        video_writer_agentview = VideoWriter(os.path.join(video_folder, "agentview"), save_video=True,
                                             single_video=False)
        video_writer_wristcameraview = VideoWriter(os.path.join(video_folder, "wristcameraview"), save_video=True,
                                                   single_video=False)

        # yy: env preparation
        env_args = {
            "bddl_file_name": [
                os.path.join(
                    cfg_.bddl_folder,
                    task_ls[i].problem_folder,
                    task_ls[i].bddl_file
                )
                for i, cfg_ in enumerate(cfg_ls)
            ],
            "camera_heights": [cfg_.data.img_h for _, cfg_ in enumerate(cfg_ls)],
            "camera_widths": [cfg_.data.img_w for _, cfg_ in enumerate(cfg_ls)],
        }
        env_num = cfg['eval']['n_eval']
        env = SubprocVectorEnv(
            [
                lambda: SequentialEnv(n_tasks=len(cfg_ls), init_states_ls=init_states_ls, **env_args)
                for _ in range(env_num)
            ]
        )
        env.reset()
        env.seed(cfg.seed)
        algo.reset()
        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        indices = np.arange(env_num) % init_states.shape[0]
        init_states_ = init_states[indices]
        obs = env.set_init_state(init_states_)
        dones = [False] * env_num
        task_indexes = [0 for _ in range(env_num)]
        steps = 0
        num_success = 0
        level_success_rate = {int(task_idx): 0 for task_idx in range(n_tasks)}
        for _ in range(5):  # simulate the physics without any actions
            env.step(np.zeros((env_num, 7)))

        # TODO: Start coding from this line!!!
        # yy: formal start of the evaluation
        with torch.no_grad():
            while steps < (cfg.eval.max_steps * n_tasks):

                t_0 = time.time()

                steps += 1
                if steps % (cfg.eval.max_steps // 30) == 0:
                    print(f"[INFO] Steps: {steps}; Task Indexes: {task_indexes}.", flush=True)
                    print(f"Evaluation takes {t.get_middle_past_time()} seconds", flush=True)
                #
                #
                #
                #
                #
                # # # Initialize an empty list to store actions
                # # actions_list = []
                # # # Prepare data for all tasks in a single loop
                # #
                # # t_1 = time.time()
                # # for k in range(env_num):
                # #     task_emb = benchmark.get_task_emb(task_idx_ls[task_indexes[k]])
                # #     cfg = cfg_ls[task_indexes[k]]
                # #     algo = algo_ls[task_indexes[k]]
                # #     if k == 0:
                # #         t_1_1 = time.time()
                # #     # Convert observations to tensor format
                # #     data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                # #     if k == 0:
                # #         t_1_2 = time.time()
                # #     # Prepare data for the k'th value
                # #     for key, v in data['obs'].items():
                # #         data['obs'][key] = v[k, ...][None, ...]
                # #     data['task_emb'] = data['task_emb'][k, ...][None, ...]
                # #     # Collect data for policy action retrieval
                # #     actions_list.append(data)
                # #     if k == 0:
                # #         t_1_3 = time.time()
                #
                #
                #
                # actions_list = []
                # cfgs = []
                #
                # for k in range(env_num):
                #     task_emb = benchmark.get_task_emb(task_idx_ls[task_indexes[k]])
                #     task_embs.append(task_emb)
                #     cfgs.append(cfg_ls[task_indexes[k]])
                #
                # # Convert observations to tensor format using all gathered task embeddings and configurations
                # data = raw_obs_to_tensor_obs(obs, torch.stack(task_embs), cfgs[0])
                #
                # for k in range(env_num):
                #     data_cp = data.copy()
                #     # Prepare data for the k'th value
                #     for key, v in data['obs'].items():
                #         data_cp['obs'][key] = v[k, ...][None, ...]
                #     data_cp['task_emb'] = data_cp['task_emb'][k, ...][None, ...]
                #     # Collect data for policy action retrieval
                #     actions_list.append(data_cp)
                #     if k == 0:
                #         t_1_3 = time.time()
                #
                #
                # # # Initialize an empty list to store actions
                # # actions_list = []
                # #
                # # # Gather all task embeddings and configurations for the current observations
                # # task_embs = []
                # # cfgs = []
                # #
                # # for k in range(env_num):
                # #     task_emb = benchmark.get_task_emb(task_idx_ls[task_indexes[k]])
                # #     task_embs.append(task_emb)
                # #     cfgs.append(cfg_ls[task_indexes[k]])
                # #
                # # # Convert observations to tensor format using all gathered task embeddings and configurations
                # # t_1_1 = time.time()
                # # data = raw_obs_to_tensor_obs(obs, torch.stack(task_embs), cfgs[0])  # Call once with stacked task embeddings
                # # t_1_2 = time.time()
                # #
                # # # Prepare data for each k'th value and collect data for policy action retrieval
                # # for k in range(env_num):
                # #     # Prepare data for the k'th value
                # #     action_data = {}
                # #     for key in data['obs']:
                # #         # Ensure we extract the observation corresponding to k
                # #         action_data['obs'] = {key: data['obs'][key][k, ...][None, ...]}  # Get k-th observation
                # #     action_data['task_emb'] = data['task_emb'][k, ...][None, ...]  # Get corresponding task embedding
                # #     actions_list.append(action_data)  # Append the prepared action data
                # #
                # # t_1_3 = time.time()
                #
                #
                #
                # # Stack all observations and task embeddings using PyTorch
                # all_obs = {key: torch.cat([d['obs'][key] for d in actions_list], dim=0) for key in data['obs'].keys()}
                # all_task_emb = torch.cat([d['task_emb'] for d in actions_list], dim=0)
                #
                # t_3 = time.time()
                #
                # # Call policy once with all data
                # actions = algo.policy.get_action({'obs': all_obs, 'task_emb': all_task_emb})
                #
                # t_4 = time.time()
                #
                # # Step the environment with the actions
                # obs, reward, done, info = env.step(actions)
                # task_indexes = [kv['task_index'] for kv in info]
                #
                # t_5 = time.time()


                # actions = np.zeros((1, 7))
                # for k in range(env_num):
                #     task_emb = benchmark.get_task_emb(task_idx_ls[task_indexes[k]])
                #     cfg = cfg_ls[task_indexes[k]]
                #     algo = algo_ls[task_indexes[k]]
                #     data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                #     # only take the k'th value for data
                #     for key, v in data['obs'].items():
                #         data['obs'][key] = v[k, ...][None, ...]
                #     data['task_emb'] = data['task_emb'][k, ...][None, ...]
                #     actions = np.vstack([actions, algo.policy.get_action(data)])
                # actions = actions[1:, ...]
                # obs, reward, done, info = env.step(actions)
                # task_indexes = [kv['task_index'] for kv in info]

                actions = np.zeros((1, 7))
                # For the 20 envs, each may have different language descriptions, i.e., task_emb
                task_embs = []
                for k in range(env_num):
                    task_emb = benchmark.get_task_emb(task_idx_ls[task_indexes[k]])
                    task_embs.append(task_emb)
                task_embs = torch.stack(task_embs)
                t_1 = time.time()
                # Obtain torch data - main inputs: obs + task_embs
                data = raw_obs_to_tensor_obs(obs, task_embs, cfg, is_sequential_env=True)
                t_2 = time.time()
                for k in range(env_num):
                    data_cp = copy.deepcopy(data)
                    algo = algo_ls[task_indexes[k]]
                    # only take the k'th value for data
                    for key, v in data_cp['obs'].items():
                        data_cp['obs'][key] = v[k, ...][None, ...]
                    data_cp['task_emb'] = data_cp['task_emb'][k, ...][None, ...]
                    actions = np.vstack([actions, algo.policy.get_action(data_cp)])
                actions = actions[1:, ...]
                obs, reward, done, info = env.step(actions)
                task_indexes = [kv['task_index'] for kv in info]


                # yy: reset robot arm if move to a new skill. Modify the obs as well.
                obs = reset_env_init_states(env, obs, info, init_states_ls, env_num, task_indexes)

                t_3 = time.time()

                video_writer_agentview.append_vector_obs(
                    obs, dones, camera_name="agentview_image"
                )
                video_writer_wristcameraview.append_vector_obs(
                    obs, dones, camera_name="robot0_eye_in_hand_image"
                )

                # check whether succeed
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]
                if all(dones):
                    break

                print(f"t_1 - t_0: {t_1 - t_0}")
                print(f"t_2 - t_1: {t_2 - t_1}")
                print(f"t_3 - t_2: {t_3 - t_2}")

                exit(0)

            for k in range(env_num):
                num_success += int(dones[k])

            """
            level_info
            """
            level_info = np.array([kv['complete_id'] for kv in info])
            for level, succ_ls in level_success_rate.items():
                level_success_rate[level] = np.sum(level_info >= level) / env_num

        video_writer_agentview.save(save_video_name="video_agentview")
        video_writer_wristcameraview.save(save_video_name="video_wristcameraview")
        success_rate = num_success / env_num
        env.close()

        eval_stats = {
            "success_rate": success_rate,
            "level_success_rate": level_success_rate
        }

        torch.save(eval_stats, save_stats_pth)

    with open(os.path.join(save_dir, f"succ_rate_evaluation_on_ori_envs.npy"), 'wb') as f:
        np.save(f, success_rate)
    with open(os.path.join(save_dir, f"level_succ.pkl"), 'wb') as f:
        pickle.dump(level_success_rate, f)

    print(
        f"[info] finish for ckpt at {model_path} in {t.get_elapsed_time()} sec for rollouts"
    )
    print(f"Results are saved at {save_stats_pth}")
    print(success_rate)
    eval_task_id.append(task_id)

    print(f"[INFO] Finish evaluating original env list: {eval_task_id}")


if __name__ == "__main__":
    main()
