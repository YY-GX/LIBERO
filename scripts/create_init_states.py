import torch
from libero.libero.envs import OffScreenRenderEnv
import os

import os
suite_name = "yy_try"
# bddl_file_name = f"/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/{suite_name}/SHELF_TABLE_SCENE_moving_potato_from_shelf_to_the_plate_on_the_table.bddl"
# out_file = f"/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/init_files/{suite_name}"

bddl_file_name = f"/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/{suite_name}/KITCHEN_SCENE_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.bddl"
out_file = f"/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/init_files/{suite_name}"
out_file = os.path.join(out_file, bddl_file_name.split(suite_name)[-1].replace(".bddl", ".pruned_init").lstrip('/'))

env_args = {
            "bddl_file_name": bddl_file_name,
            "camera_heights": 128,
            "camera_widths": 128,
        }
env = OffScreenRenderEnv(**env_args)
dim = env.get_sim_state().shape[0]
N = 3
env.reset()
init_states = torch.from_numpy(env.get_sim_state().reshape((1, dim)))
for _ in range(N - 1):
    env.reset()
    init_states = torch.vstack([init_states, torch.from_numpy(env.get_sim_state().reshape((1, dim)))])
# print(init_states.size())
# print(init_states)
torch.save(init_states, out_file)