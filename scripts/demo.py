import torch
import numpy as np
from os.path import join, dirname
from PIL import Image
import cv2
import argparse

import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from usam.patch_sam2 import patch_sam2


def load_args():
    # Define paths
    root = __file__.split("scripts")[0]
    sam2_root = dirname(sam2.__file__)

    parser = argparse.ArgumentParser(description="Demo for SAM2 with MLP")
    parser.add_argument(
        "--model_cfg",
        type=str,
        default=join(sam2_root, "configs", "sam2", "sam2_hiera_t.yaml"),
        help="Path to the model config")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=join(root, "models", "sam", "checkpoints_2.0", "sam2_hiera_tiny.pt"),
        help="Path to the checkpoint")
    parser.add_argument(
        "--mlp-dir",
        type=str,
        default=join(root, "models", "mlps", "sam2.0"),
        help="Path to the MLP directory")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera ID for live inference")

    return parser.parse_args()


def load_predictor(model_cfg, checkpoint, mlp_dir):
    # Load predictor
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    patch_sam2(predictor, mlp_dir)
    return predictor


def show_image(image, mask, mlp_scores):
    """
    Visualizes an image with crosshairs in the middle of the image, a mask
    verlay, and MLP scores visualized in the bottom left corner. Furthermore,
    there is a cool styled headline in the top center that says: "CertainSAM".

    :param image:
    :param mask:
    :param mlp_scores:
    :return:
    """

    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image[mask == 1] = 0.5 * np.asarray([0, 255, 0]) + 0.5 * image[mask == 1]

    # Draw crosshairs
    center = (image.shape[1] // 2, image.shape[0] // 2)
    cv2.line(image, (center[0], max(0, center[1]-30)), (center[0], min(2 *center[1], center[1]+30)), (255, 0, 0), 1)
    cv2.line(image, (max(0, center[0]-30), center[1]), (max(0, center[0]+30), center[1]), (255, 0, 0), 1)
    # Draw a circle around the crosshairs
    cv2.circle(image, center, 15, (255, 0, 0), 1)


    # Draw MLP scores with a gray transparent background
    cv2.rectangle(image, (0, 0), (200, 100), (0, 0, 0), -1)
    for i, (key, score) in enumerate(mlp_scores.items()):
        cv2.putText(image, f"{key}: {float(score):.2f}", (10, 30 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw headline
    cv2.putText(image, "CertainSAM", (256, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("image", image)
    key = cv2.waitKey(1)
    return key


def inference_loop(predictor, camera):
    # Open Camera
    cap = cv2.VideoCapture(camera)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            input_point = np.array([[frame.shape[1]//2, frame.shape[0]//2]])
            input_label = np.array([1])

            predictor.set_image(frame)
            masks, scores, logits, mlp_scores = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )
            sorted_ind = np.argsort(scores)[::-1]
            mask = masks[sorted_ind][0]
            mlp_scores = {k: v[sorted_ind][0] for k, v in mlp_scores.items()}
            mlp_scores = {
                "Model Gap": mlp_scores["model_gap"],
                "Prompt Gap": mlp_scores["prompt_gap"],
                "Task Gap": mlp_scores["task_gap"],
            }

            key = show_image(frame, mask, mlp_scores)
            if key & 0xFF == ord("q"):
                break

def main():
    args = load_args()
    predictor = load_predictor(args.model_cfg, args.checkpoint, args.mlp_dir)
    inference_loop(predictor, args.camera)


if __name__ == "__main__":
    main()
