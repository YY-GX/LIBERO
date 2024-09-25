import numpy as np
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
import matplotlib.pyplot as plt

modified_bddl_pth = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/modified_libero/KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_with_bottom_drawer_open.bddl"
ori_bddl_pth = "/home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/libero_90/KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.bddl"

# Set up environment
env_args = {
    "bddl_file_name": ori_bddl_pth,
    "camera_heights": 512,
    "camera_widths": 512,
}
env = OffScreenRenderEnv(**env_args)
obs = env.reset()
Image.fromarray(obs["agentview_image"][::-1]).save("/home/yygx/UNC_Research/pkgs_simu/LIBERO/sam/try_imgs/ori_imgs/ori_img.png")


# Set up environment
env_args = {
    "bddl_file_name": modified_bddl_pth,
    "camera_heights": 512,
    "camera_widths": 512,
}
env = OffScreenRenderEnv(**env_args)
obs = env.reset()
Image.fromarray(obs["agentview_image"][::-1]).save("/home/yygx/UNC_Research/pkgs_simu/LIBERO/sam/try_imgs/modified_imgs/modified_img.png")
