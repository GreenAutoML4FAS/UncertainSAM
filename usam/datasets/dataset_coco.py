from os.path import join
from PIL import Image
import numpy as np
import json
from pycocotools.coco import COCO

from usam.datasets.dataset import ImageDataset


class COCOImageDataset(ImageDataset):
    def __init__(
            self,
            root,
            **kwargs,
    ):

        super(COCOImageDataset, self).__init__(root, **kwargs)

    def __parse__(self):
        # Load video name file
        if self.train:
            self.annotation_file = "annotations/instances_train2017.json"
            self.image_subdir = "train2017"
        elif self.validation:
            self.annotation_file = "annotations/instances_val2017.json"
            self.image_subdir = "val2017"
        else:
            self.annotation_file = None
            self.image_subdir = None

        # Read annotation file
        with open(join(self.root, self.annotation_file), "r") as file:
            self.annotation = json.load(file)

        self.annotations = list()
        self.video_names = list()
        self.video_paths = list()

        annotations = dict()
        for ann in self.annotation["annotations"]:
            if ann["image_id"] not in annotations:
                annotations[ann["image_id"]] = list()
            annotations[ann["image_id"]].append(ann)

        for img in self.annotation["images"]:
            if img["id"] not in annotations:
                continue
            self.annotations.append(annotations[img["id"]])
            self.video_names.append(img["file_name"])
            self.video_paths.append(join(self.root, self.image_subdir, img["file_name"]))

        # Load video lengths
        self.video_lengths = [1 for _ in self.video_paths]
        self.annotation = COCO(join(self.root, self.annotation_file))


    def __load_sample__(self, idx):
        # Find video index
        video_idx = self.__find_video_idx__(idx)
        # Load image
        relative_image_name = join(self.image_subdir, self.video_names[video_idx])
        image_path = join(self.video_paths[video_idx])
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))

        annotations = self.annotations[video_idx]
        targets = dict()
        for i, ann in enumerate(annotations):
            if ann["iscrowd"]:
                continue
            if ann["area"] < 10:
                continue
            mask = self.annotation.annToMask(ann)
            targets[int(i)] = mask

        return image, targets, relative_image_name


    def create_relative_image_name(self, idx):
        video_idx = self.__find_video_idx__(idx)
        # Load image
        relative_image_name = join(self.image_subdir, self.video_names[video_idx])
        return relative_image_name


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser(
        description="Cityscapes Image Dataset"
    )
    PARSER.add_argument(
        "--root", required=True, help="Path to the root of the dataset"
    )

    ARGS = PARSER.parse_args()

    dataset = COCOImageDataset(ARGS.root, split="val")
    print(len(dataset))
    image, targets, centers, sample_points, image_name = dataset[0]
    print(image.shape, targets.keys(), image_name)
