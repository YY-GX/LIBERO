from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np

img = np.array(cv2.imread("./cabinet_img.png"))
sam = sam_model_registry["default"](checkpoint="./sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
predictor.set_image(img)
masks, _, _ = predictor.predict("wooden_cabinet")