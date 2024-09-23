import os

from libero.libero.envs import OffScreenRenderEnv
import h5py
import numpy as np
from PIL import Image

def hdf5_to_dict(group):
    result = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):  # If the item is a group, recursively convert it
            result[key] = hdf5_to_dict(item)
        elif isinstance(item, h5py.Dataset):  # If the item is a dataset, convert it to a numpy array
            result[key] = item[()]
        else:
            raise TypeError(f"Unsupported HDF5 item type: {type(item)}")
    return result

def load_hdf5_file_to_dict(file_path):
    with h5py.File(file_path, 'r') as f:
        return hdf5_to_dict(f)


img_saved_folder = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/sam/try_imgs/wrist_imgs"
img_size = 512

env_args = {
    "bddl_file_name": "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/modified_libero/KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle_with_popcorn_in_the_bowl.bddl",
    "camera_heights": img_size,
    "camera_widths": img_size,
}
env = OffScreenRenderEnv(**env_args)

demo_pth = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/datasets/libero_90/KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle_demo.hdf5"
data_dict = load_hdf5_file_to_dict(demo_pth)['data']
for demo_idx in list(data_dict.keys()):
    print(f">> demo_idx: {demo_idx}")
    demo = data_dict[demo_idx]
    actions = demo['actions']
    for i, action in enumerate(actions):
        print(f">> Steps: {i}")
        obs, _, _, _ = env.step(action)
        image = Image.fromarray(obs['robot0_eye_in_hand_image'])
        image.save(os.path.join(img_saved_folder, f"demo_{demo_idx}_wrist_idx{i}.png"))
    env.reset()
    exit(0)

env.close()
