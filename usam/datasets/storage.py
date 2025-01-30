from os import makedirs
from os.path import join, exists, dirname
import gc

import pickle

from pycocotools import mask as mask_util


class Metrics:
    """
    Class to store the metrics corresponding to a prediction set
    """
    def __init__(
            self,
            best_iou=None,
            unsupervised_iou=None,
            sam_score=None,
            unsupervised_sam_score=None,
            entropy=None,
            iou_dense_prompt=None,
            h_all=None,
            h_theta=None,
            h_task=None,
            h_prompt=None,
    ):
        # Standard metrics
        self.iou = best_iou
        self.unsupervised_iou = unsupervised_iou
        self.sam_score = sam_score
        self.unsupervised_sam_score = unsupervised_sam_score
        self.entropy = entropy
        self.iou_dense_prompt = iou_dense_prompt
        self.h_all = h_all
        self.h_theta = h_theta
        self.h_task = h_task
        self.h_prompt = h_prompt


class Prediction:
    """
    Class to store the prediction result for an image and a prompt
    """
    def __init__(
            self,
            scores,
            iou_token,
            mask_token,
            metrics
    ):
        self.scores = scores
        self.iou_token = iou_token
        self.mask_token = mask_token
        self.metrics = metrics


class Storage:
    """
    Class to store an entire set of predictions with the corresponding metrics,
    meta information, and the applied augmentation
    """
    def __init__(
            self,
            root: str,
            file_name: str,
    ):
        if file_name.endswith(".pkl"):
            file_name = file_name.replace(".pkl", "")
        if file_name.startswith("\\"):
            file_name = file_name[1:]

        self.root = root
        self.file_name = file_name
        self.content = None

        if self.__exist__():
            self.content = self.__load__()

        if self.content is None:
            self.content = self.__create__()

    def __load__(self):
        gc.disable()
        try:
            with open(join(self.root, self.file_name+".pkl"), "rb") as file:
                res = pickle.load(file)
        except Exception as e:
            print(join(self.root, self.file_name+".pkl"), e)
            res = None

        gc.enable()
        return res

    def save(self):
        path = dirname(join(self.root, self.file_name+".pkl"))
        if not exists(path):
            makedirs(path, exist_ok=True)
        with open(join(self.root, self.file_name+".pkl"), "wb+") as file:
            pickle.dump(self.content, file)

    def __exist__(self):
        return exists(join(self.root, self.file_name+".pkl"))

    def __create__(self):
        return {
            "image_name": self.file_name,
            "image_width": None,
            "image_height": None,
            "ground_truth_mask_sizes": {},  # {id: area}
            "prediction": {},  # {model_name: {id: Prediction}}
            "augmentations": {},  # {model_name: {id: {aug: Prediction}}}
        }

    def set_image_name(self, image_name):
        self.content["image_name"] = image_name

    def set_image_width(self, image_width):
        self.content["image_width"] = image_width

    def set_image_height(self, image_height):
        self.content["image_height"] = image_height

    def set_ground_truth_mask_sizes(self, id, area):
        self.content["ground_truth_mask_sizes"][id] = area

    def set_prediction(
            self,
            model,
            idx,
            scores,
            iou_token,
            mask_token,
            metrics,
            augmentation=None
    ):
        p = Prediction(scores, iou_token, mask_token, metrics)
        if not augmentation:
            if model not in self.content["prediction"]:
                self.content["prediction"][model] = {}
            self.content["prediction"][model][idx] = p
        else:
            if model not in self.content["augmentations"]:
                self.content["augmentations"][model] = {}
            if idx not in self.content["augmentations"][model]:
                self.content["augmentations"][model][idx] = {}
            self.content["augmentations"][model][idx][augmentation] = p

    def get_ids(self):
        return list(self.content["ground_truth_mask_sizes"].keys())

    def get_models(self):
        return list(self.content["prediction"].keys())

    def get_augmentations(self, model, idx):
        return self.content["augmentations"][model][idx]

    def get_prediction(self, model_name, idx, augmentation=None) -> Prediction:
        if augmentation is None:
            return self.content["prediction"][model_name][idx]
        return self.content["augmentations"][model_name][idx][augmentation]

    def get_ground_truth_mask_sizes(self, id):
        return self.content["ground_truth_mask_sizes"][id]

