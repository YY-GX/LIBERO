import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import groundingdino
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict, annotate
from PIL import Image

import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid



def load_image(image):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(Image.fromarray(image), None)
    return image, image_transformed



# --------------------------------------------------------------------------------------------------------------------

def obtain_mask(
        img,
        text_prompt,
        points_prompt,
):
    """

    Input:
        img: numpy arr - [512, 512, 3]
        points_prompt: numpy arr N*2 - [[p_x_1, p_y_1], [p_x_2, p_y_2]]
        text_prompt: str
    Return:
        mask
    """

    # VERY important: text queries need to be lowercased + end with a dot
    TEXT_PROMPT = text_prompt
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25

    SAM2_CHECKPOINT = "/mnt/arc/yygx/pkgs_baselines/Grounded-SAM-2/checkpoints/sam2_hiera_large.pt"
    SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"
    GROUNDING_DINO_CONFIG = "/mnt/arc/yygx/pkgs_baselines/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "/mnt/arc/yygx/pkgs_baselines/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # build SAM2 image predictor
    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )

    # setup the input image and text prompt for SAM 2 and Grounding DINO
    text = TEXT_PROMPT
    image_source, image = load_image(img)
    sam2_predictor.set_image(image_source)

    if text:
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=text,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        # process the box prompt for SAM 2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    if points_prompt:
        """
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
        """
        masks, scores, logits = sam2_predictor.predict(
            point_coords=points_prompt,
            point_labels=None,
            box=None,
            multimask_output=False,
        )

    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = confidences.numpy().tolist()
    max_confidence_index = np.argmax(confidences)
    best_mask = masks[max_confidence_index]

    return best_mask



def inpainting(
       img,
       mask_img,
       prompt,
       negative_prompt
):
    """
    Input:
        img [512, 512, 3]
        mask_img [512, 512]
        prompt: str
        text_prompt: str
    Return:
        img [512, 512, 3]
    """
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
    )
    pipeline.enable_model_cpu_offload()
    # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    # pipeline.enable_xformers_memory_efficient_attention()
    print(type(img), img.shape)
    print(type(mask_img), mask_img.shape)
    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=img, mask_image=mask_img).images[0]
    output_dir = "/mnt/arc/yygx/pkgs_baselines/LIBERO/sam/outputs_test/inpaint_img.png"
    image.save(output_dir)
    image = np.array(image)
    print(type(image), image.shape)
    return image




def add_ori_obj(
        img,
        obj_img
):
    """
    Input:
        img [512, 512, 3]
        obj_img [512, 512, 3]
    Return:
        img [128, 128, 3]
    """
    pass

if __name__ == "__main__":
    img_path = "/mnt/arc/yygx/pkgs_baselines/LIBERO/sam/try_imgs/wrist_imgs/demo_demo_0_wrist_idx54.png"
    img_path = "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/experiments/libero_90/training_eval_skills_original_env/Sequential/BCViLTPolicy_seed10000/all/eval_tasks_on_modified_envs_seed10000/evaluation_task1_on_modified_envs/load_ori_1_on_modified_1_videos/agentview/video_agentview_0/high_res_v3.png"
    img, _ = groundingdino.util.inference.load_image(img_path)
    mask = obtain_mask(
        img,
        text_prompt="popcorn box.",
        points_prompt=None
    )
    print(mask.shape)
    img = img[:2048, :2048]
    mask = mask[:2048, :2048]
    img = inpainting(
        img=img,
        mask_img=mask,
        prompt="complete the image",
        negative_prompt="bad anatomy, deformed, ugly, disfigured"
    )
    print(img.shape)
