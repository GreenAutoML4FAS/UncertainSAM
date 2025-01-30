from os.path import join, exists
from PIL import Image
import numpy as np
import json

from usam.datasets.dataset import ImageDataset


class MOSEImageDataset(ImageDataset):
    def __init__(
            self,
            root: str,
            split: str = "train",
            **kwargs,
    ):
        self.prefix = None

        super(MOSEImageDataset, self).__init__(
            root, split, **kwargs)

    def __parse__(self):
        # Set prefix
        if self.train:
            self.prefix = "train"
        elif self.validation:
            self.prefix = "valid"
        else:
            raise NotImplementedError("Test set is not available for MOSE")
        # Read in json file
        video_meta_file = join(self.root, f"meta_{self.prefix}.json")
        with open(video_meta_file, "r") as file:
            meta = json.load(file)

        for name, data in meta["videos"].items():
            self.video_names.append(name)
            self.video_lengths.append(data["length"])

        # Load video paths
        self.video_paths = [
            join(self.root, self.prefix, "JPEGImages", name)
            for name in self.video_names
        ]


    def __load_sample__(self, idx):
        # Find video index
        video_idx = self.__find_video_idx__(idx)
        # Get frame idx
        frame_idx = self.__find_frame_idx__(idx)
        # Load image
        relative_image_name = join(
            self.prefix, "JPEGImages", self.video_names[video_idx],
            f"{frame_idx:05d}.jpg"
        )
        image_path = join(self.root, relative_image_name)
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))

        targets = None
        if self.train or (self.validation and frame_idx == 0):
            target_path = join(
                self.root, self.prefix, "Annotations",
                self.video_names[video_idx], f"{frame_idx:05d}.png"
            )
            if exists(target_path):
                target = Image.open(target_path)
                target = np.array(target)
                targets = dict()
                for i in np.unique(target):
                    if i != 0:
                        targets[int(i)] = target == i

        return image, targets, relative_image_name


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser(
        description="MOSE Image Dataset"
    )
    PARSER.add_argument(
        "--root", required=True, help="Path to the root of the dataset"
    )

    ARGS = PARSER.parse_args()

    dataset = MOSEImageDataset(ARGS.root, split="train")
    print(len(dataset))
    image, target, image_name = dataset[0]
    print(image.shape, image_name)

