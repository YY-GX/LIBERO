import os
import numpy as np
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
import matplotlib.pyplot as plt

base_ori_path = "/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/libero_90"
base_modified_path = "/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/bddl_files/modified_libero"

dst_base_ori_path = "/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/first_frames/ori"
dst_base_modified_path = "/home/yygx/Dropbox/Codes/UNC_Research/pkgs_simu/LIBERO/libero/libero/first_frames/modified"

modified_mapping = {
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet": [
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_with_bottom_drawer_open.bddl"],
    "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet": [
        "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_with_bottom_drawer_open.bddl",
        "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_with_popcorn_in_the_top_drawer.bddl"],
    "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet": [
        "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_with_top_drawer_open.bddl"],
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet": [
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_with_bottom_drawer_open.bddl"],
    "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate": [
        "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_with_popcorn_in_the_bowl.bddl"],
    "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet": [
        "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet_with_popcorn_in_the_bowl.bddl"],
    "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet": [
        "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet_with_the_bottom_drawer_open.bddl"],
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate": [
        "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate_with_popcorn_in_the_bowl.bddl"],
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate": [
        "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate_with_popcorn_in_the_bowl.bddl"],
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate": [
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate_with_popcorn_in_the_bowl.bddl"],

    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet": [
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet_with_top_drawer_open.bddl",
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet_with_popcorn_on_top_of_the_cabinet.bddl",
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet_with_popcorn_in_the_bowl.bddl"],
    "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle": [
        "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle_with_popcorn_in_the_bowl.bddl"],
    "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl": [
        "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl_with_popcorn_in_the_bowl.bddl"],
    "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove": [
        "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove_with_popcorn_in_the_frying_pan.bddl"],
    "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove": [
        "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove_with_stove_turned_on.bddl"],
    "KITCHEN_SCENE3_turn_on_the_stove": [
        "KITCHEN_SCENE3_turn_on_the_stove_with_moka_pot_on_the_stove.bddl"],
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet": [
        "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_with_top_drawer_open.bddl"],
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet": [
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_with_popcorn_in_the_bottom_drawer.bddl"],
    "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet": [
        "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet_with_popcorn_in_the_balck_bowl.bddl"],
    "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet": [
        "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet_with_popcorn_in_the_drawer.bddl"],
}

for k, v in modified_mapping.items():
    ori_bddl_pth = os.path.join(base_ori_path, k+".bddl")
    dst_img_ori_path = os.path.join(dst_base_ori_path, k+".png")
    env_args = {
        "bddl_file_name": ori_bddl_pth,
        "camera_heights": 512,
        "camera_widths": 512,
    }
    env = OffScreenRenderEnv(**env_args)
    obs = env.reset()
    for _ in range(5):  # simulate the physics without any actions
        obs, _, _, _ = env.step(np.zeros(7))
    Image.fromarray(obs["agentview_image"][::-1]).save(dst_img_ori_path)

    for modified_bddl in v:
        modified_bddl_pth = os.path.join(base_modified_path, modified_bddl)
        dst_img_modified_path = os.path.join(dst_base_modified_path, modified_bddl.split(".")[0]+".png")
        env_args = {
            "bddl_file_name": modified_bddl_pth,
            "camera_heights": 512,
            "camera_widths": 512,
        }
        env = OffScreenRenderEnv(**env_args)
        obs = env.reset()
        for _ in range(5):  # simulate the physics without any actions
            obs, _, _, _ = env.step(np.zeros(7))
        Image.fromarray(obs["agentview_image"][::-1]).save(dst_img_modified_path)
