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

# dds cloudapi for Grounding DINO 1.5
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk import DetectionTask
from dds_cloudapi_sdk import TextPrompt
from dds_cloudapi_sdk import DetectionModel
from dds_cloudapi_sdk import DetectionTarget

import pickle

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
        is_dino15=False
):
    """

    Input:
        img: numpy arr - [512, 512, 3]
        points_prompt: numpy arr N*2 - [[p_x_1, p_y_1], [p_x_2, p_y_2]]
        text_prompt: str
    Return:
        mask
    """
    # Check if both files exist
    mask_exists = os.path.exists(os.path.join(output_dir, "mask.npy"))
    debug_info_exists = os.path.exists(os.path.join(output_dir, "debug_info.pkl"))
    if mask_exists and debug_info_exists:
        mask = np.load(os.path.join(output_dir, "mask.npy"))
        with open(os.path.join(output_dir, "debug_info.pkl"), 'rb') as f:
            debug_info = pickle.load(f)
        print("Both files exist.")
        return mask, debug_info

    # VERY important: text queries need to be lowercased + end with a dot
    TEXT_PROMPT = text_prompt
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25

    SAM2_CHECKPOINT = "/mnt/arc/yygx/pkgs_baselines/Grounded-SAM-2/checkpoints/sam2_hiera_large.pt"
    SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"
    GROUNDING_DINO_CONFIG = "/mnt/arc/yygx/pkgs_baselines/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "/mnt/arc/yygx/pkgs_baselines/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if is_dino15:
        API_TOKEN = "3ada5c5a6347921ce6fec191f01a3b63"
        GROUNDING_MODEL = DetectionModel.GDino1_5_Pro

        """
        Prompt Grounding DINO 1.5 with Text for Box Prompt Generation with Cloud API
        """
        # Step 1: initialize the config
        token = API_TOKEN
        config = Config(token)

        # Step 2: initialize the client
        client = Client(config)

        # Step 3: run the task by DetectionTask class
        # image_url = "https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/detection/iron_man.jpg"
        # if you are processing local image file, upload them to DDS server to get the image url
        Image.fromarray(img).save('./tmp_img.png')
        img_path = './tmp_img.png'
        image_url = client.upload_file(img_path)

        task = DetectionTask(
            image_url=image_url,
            prompts=[TextPrompt(text=TEXT_PROMPT)],
            targets=[DetectionTarget.BBox],  # detect bbox
            model=GROUNDING_MODEL,  # detect with GroundingDino-1.5-Pro model
        )

        client.run_task(task)
        result = task.result

        objects = result.objects  # the list of detected objects

        input_boxes = []
        confidences = []
        class_names = []

        for idx, obj in enumerate(objects):
            input_boxes.append(obj.bbox)
            confidences.append(obj.score)
            class_names.append(obj.category)

        input_boxes = np.array(input_boxes)

        """
        Init SAM 2 Model and Predict Mask with Box Prompt
        """

        # environment settings
        # use bfloat16
        torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # build SAM2 image predictor
        sam2_checkpoint = SAM2_CHECKPOINT
        model_cfg = SAM2_MODEL_CONFIG
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
        sam2_predictor = SAM2ImagePredictor(sam2_model)

        image = Image.open(img_path)

        sam2_predictor.set_image(np.array(image.convert("RGB")))

        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        confidences = np.array(confidences)

    else:
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

    debug_info = [input_boxes, masks, confidences]
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = confidences.tolist()
    max_confidence_index = np.argmax(confidences)
    best_mask = masks[max_confidence_index]

    np.save(os.path.join(output_dir, "mask.npy"), best_mask)
    with open(os.path.join(output_dir, "debug_info.pkl"), 'wb') as f:
        pickle.dump(debug_info, f)

    return best_mask, debug_info



def inpainting(
       img,
       mask_img,
       prompt,
       negative_prompt,
       output_dir
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


def visualize_mask(
        img,
        input_boxes,
        masks,
        confidences,
        OUTPUT_DIR

):
    # input_boxes = np.expand_dims(input_boxes, 0)
    # masks = np.expand_dims(masks, 0)
    # labels = [labels]
    # confidences = [confidences]

    """
    Visualize image with supervision useful API
    """
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=np.array([i for i in range(masks.shape[0])])
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    # label_annotator = sv.LabelAnnotator()
    # annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    # cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)


if __name__ == "__main__":
    img_path = "/mnt/arc/yygx/pkgs_baselines/LIBERO/sam/try_imgs/wrist_imgs/demo_demo_0_wrist_idx54.png"
    img_path = "/mnt/arc/yygx/pkgs_baselines/LIBERO/sam/try_imgs/agent_imgs/demo_demo_0_idx0.png"
    output_dir = "/mnt/arc/yygx/pkgs_baselines/LIBERO/sam/outputs_test/"
    text_prompt = "black drawer"

    img, _ = groundingdino.util.inference.load_image(img_path)
    mask, debug_info = obtain_mask(
        img,
        text_prompt=text_prompt,
        points_prompt=None,
        is_dino15=True
    )
    print(mask.shape)

    # visualize
    input_boxes, masks, confidences = debug_info
    print(len(input_boxes))
    print(confidences)
    visualize_mask(
        img,
        input_boxes,
        masks,
        confidences,
        output_dir
    )

    inpaint_prompt = "Fill the masked area to match the texture and color of the surrounding environment."
    inpaint_file_name = "_".join(text_prompt.split(" ")) + "___" + "_".join(inpaint_prompt.split(" "))
    img = inpainting(
        img=img,
        mask_img=mask,
        prompt=inpaint_prompt,
        negative_prompt="bad anatomy, deformed, ugly, disfigured",
        output_dir=os.path.join(output_dir, f"inpaint_img_{inpaint_file_name}.png")
    )
    print(img.shape)
