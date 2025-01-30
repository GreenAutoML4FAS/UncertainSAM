from os import listdir
from os.path import join, exists
from PIL import Image
import numpy as np

from usam.datasets.dataset import ImageDataset


class DavisImageDataset(ImageDataset):
    def __init__(
            self,
            root: str,
            split: str = "train",
            year: int = 2017,
            challenge_dev: str = "dev",
            resolution: str = "Full-Resolution",
            video_subset: list = None,
            **kwargs,
    ):
        assert year in [2016, 2017], "Year must be 2016 or 2017"
        assert challenge_dev in ["dev",
                                 "test"], "Challenge dev must be dev or test"
        assert resolution in ["Full-Resolution",
                              "480p"], "Resolution must be Full-Resolution or 480p"
        self.year = year
        self.challenge_dev = challenge_dev
        self.resolution = resolution

        super(DavisImageDataset, self).__init__(
            root, split, video_subset=video_subset, **kwargs)

    def __parse__(self):
        # Load video name file
        if self.train:
            video_name_file = join(
                self.root, "ImageSets", str(self.year), "train.txt")
        elif self.validation:
            video_name_file = join(
                self.root, "ImageSets", str(self.year), "val.txt")
        else:
            assert self.year == 2017, "Test set is only available for 2017"
            video_name_file = join(
                self.root, "ImageSets", str(self.year),
                f"test-{self.challenge_dev}.txt"
            )

        with open(video_name_file, "r") as file:
            if self.video_subset is not None:
                for i, line in enumerate(file.read().splitlines()):
                    if i in self.video_subset:
                        self.video_names.append(line.strip())
            else:
                self.video_names = file.read().splitlines()
        self.video_names.sort()

        # Load video paths
        self.video_paths = [
            join(self.root, "JPEGImages", str(self.resolution), name)
            for name in self.video_names
        ]

        # Load video lengths
        self.video_lengths = [
            len(listdir(path)) for path in self.video_paths
        ]
        if self.test:
            self.video_lengths = [1 for _ in self.video_lengths]

    def create_relative_image_name(self, idx):
        video_idx = self.__find_video_idx__(idx)
        # Get frame idx
        frame_idx = self.__find_frame_idx__(idx)
        # Load image
        relative_image_name = join(
            "JPEGImages", str(self.resolution), self.video_names[video_idx],
            f"{frame_idx:05d}.jpg"
        )
        return relative_image_name

    def __load_sample__(self, idx):
        # Find video index
        video_idx = self.__find_video_idx__(idx)
        # Get frame idx
        frame_idx = self.__find_frame_idx__(idx)
        # Load image
        relative_image_name = join(
            "JPEGImages", str(self.resolution), self.video_names[video_idx],
            f"{frame_idx:05d}.jpg"
        )
        image_path = join(self.root, relative_image_name)
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))

        targets = None
        if self.train or (self.validation or self.test and frame_idx == 0):
            target_path = join(
                self.root, "Annotations", str(self.resolution),
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
        description="Davis Image Dataset"
    )
    PARSER.add_argument(
        "--root", required=True, help="Path to the root of the dataset"
    )

    ARGS = PARSER.parse_args()

    dataset = DavisImageDataset(ARGS.root, split="train")
    print(len(dataset))
    image, target, image_name = dataset[0]
    print(image.shape, target.keys(), image_name)
