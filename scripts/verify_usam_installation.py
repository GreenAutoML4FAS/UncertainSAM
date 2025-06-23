import torch
import numpy as np
from os.path import join, dirname
from PIL import Image
import cv2

import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from usam.patch_sam2 import patch_sam2


# Define paths
root = __file__.split("scripts")[0]
sam2_root = dirname(sam2.__file__)
model_cfg = join(sam2_root, "configs", "sam2", "sam2_hiera_t.yaml")
checkpoint = join(root, "models", "sam", "checkpoints_2.0", "sam2_hiera_tiny.pt")
mlp_dir = join(root, "models", "mlps", "sam2.0")

# Load image
image = Image.open(join(root, "docs", "assets", "dog_sample.png"))
image = np.array(image.convert("RGB"))

# Load predictor
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
patch_sam2(predictor, mlp_dir)

# Sample inference
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    input_point = np.array([[256, 256]])
    input_label = np.array([1])

    predictor.set_image(image)
    masks, scores, logits, mlp_scores = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    sorted_ind = np.argsort(scores)[::-1]
    mask = masks[sorted_ind][0]
    score = scores[sorted_ind][0]
    logit = logits[sorted_ind][0]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image[mask == 1] = 0.5 * np.asarray([0, 255, 0]) + 0.5 * image[mask == 1]

    print(f"Inferred an image with shape {image.shape}, mask shape {mask.shape}, logit shape {logit.shape}, and score {score}")
    print(f"MLP scores: {mlp_scores}")
    cv2.imshow("image", image)
    cv2.waitKey(0)
