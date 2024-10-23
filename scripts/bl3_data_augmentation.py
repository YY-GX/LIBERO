import os

from libero.libero.envs import OffScreenRenderEnv
import h5py
import numpy as np
from PIL import Image
from pathlib import Path
from robosuite.wrappers import VisualizationWrapper
from libero.libero.envs import *
from robosuite import load_controller_config
import libero.libero.envs.bddl_utils as BDDLUtils

class CreateDemos:
    def __init__(
            self,
            benchmark,
    ):
        self.benchmark = benchmark

        self.bddl_path = "/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/single_step/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_with_bowl_on_top_of_cabinet.bddl"
        self.demos_path = "/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/datasets/libero_90/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.hdf5"

        # self.img_saved_folder = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/sam/try_imgs/agent_imgs"
        # Path(self.img_saved_folder).mkdir(parents=True, exist_ok=True)

    def hdf5_to_dict(self, group):
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):  # If the item is a group, recursively convert it
                result[key] = self.hdf5_to_dict(item)
            elif isinstance(item, h5py.Dataset):  # If the item is a dataset, convert it to a numpy array
                result[key] = item[()]
            else:
                raise TypeError(f"Unsupported HDF5 item type: {type(item)}")
        return result

    def load_hdf5_file_to_dict(self, file_path):
        with h5py.File(file_path, 'r') as f:
            return self.hdf5_to_dict(f)

    def replay_demos(self):
        img_size = 512

        # env_args = {
        #     "bddl_file_name": self.bddl_path,
        #     "camera_heights": img_size,
        #     "camera_widths": img_size,
        # }
        # env = OffScreenRenderEnv(**env_args)

        controller_config = load_controller_config(default_controller="OSC_POSE")
        config = {
            "robots": "Panda",
            "controller_configs": controller_config,
            "camera_heights": img_size,
            "camera_widths": img_size,
        }

        assert os.path.exists(self.bddl_path)
        problem_info = BDDLUtils.get_problem_info(self.bddl_path)
        problem_name = problem_info["problem_name"]
        print(problem_name)
        env = TASK_MAPPING[problem_name](
            bddl_file_name=self.bddl_path,
            **config,
            has_renderer=True,
            has_offscreen_renderer=True,
            render_camera="agentview",
            ignore_done=True,
            use_camera_obs=True,
            reward_shaping=True,
            control_freq=20,
        )

        env = VisualizationWrapper(env)
        env.reset()
        env.render()
        # crr_state = env.get_sim_state()
        # ori_state = np.load("init_state.npy")
        # ori_state[38:41] = crr_state[38:41]
        # env.set_init_state(ori_state)

        demo_pth = self.demos_path
        data_dict = self.load_hdf5_file_to_dict(demo_pth)['data']
        for demo_idx in list(data_dict.keys()):
            print(f">> demo_idx: {demo_idx}")
            demo = data_dict[demo_idx]
            # np.save("init_state.npy", demo['states'][0])
            actions = demo['actions']
            for i, action in enumerate(actions):
                print(f">> Steps: {i}")
                obs, _, _, _ = env.step(action)

                # image = obs['agentview_image']
                # image = np.flip(image, axis=0)
                # image = Image.fromarray(image)
                # image.save(os.path.join(img_saved_folder, f"demo_{demo_idx}_idx{i}.png"))
            env.reset()
            exit(0)

        env.close()


if __name__ == '__main__':
    create_demos = CreateDemos(benchmark="single_step")
    create_demos.replay_demos()