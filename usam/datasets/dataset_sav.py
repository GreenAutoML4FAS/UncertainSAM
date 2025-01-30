from os import listdir
from os.path import join, exists
from PIL import Image
import numpy as np
import json
from pycocotools import mask as coco_mask
import cv2

from usam.datasets.dataset import ImageDataset


class SAVImageDataset(ImageDataset):
    def __init__(
            self,
            root: str,
            split: str = "train",
            annotation_set: str = "manual",
            train_subsets: list = None,
            **kwargs,
    ):

        # Set the list of training subsets
        if train_subsets is None:
            self.train_subsets = [int(i) for i in range(0, 56)]
        else:
            self.train_subsets = train_subsets
        assert type(self.train_subsets) == list, "train_subsets must be a list"
        assert all(isinstance(i, int) for i in self.train_subsets), \
            "train_subsets must be a list of integers"
        assert all(0 <= i < 56 for i in self.train_subsets), \
            "train_subsets must be between 0 and 55"
        assert len(self.train_subsets) > 0, "train_subsets must not be empty"
        assert len(self.train_subsets) == len(np.unique(self.train_subsets)), \
            "train_subsets must be unique"

        # Set the anotation set
        assert annotation_set in ["manual", "auto"], \
            "Annotation set must be manual or auto"
        self.annotation_set = annotation_set

        self.video_ids = []
        self.prefix = None

        super(SAVImageDataset, self).__init__(root, split, **kwargs)

    def __parse__(self):
        # Set prefix
        if self.train:
            self.prefix = "train"
        elif self.validation:
            self.prefix = "val"
        else:
            self.prefix = "test"

        # Load train data
        if self.train:
            train_root_names = []
            for i in self.train_subsets:
                train_root_names.append(f"sav_{i:03d}")

            for train_root in train_root_names:
                # Find videos with .mp4 end
                videos = []
                for video in listdir(join(self.root, f"sav_train", train_root)):
                    if video.endswith(".mp4"):
                        videos.append(video)
                videos.sort()

                for video in videos:
                    self.video_paths.append(
                        join(self.root, f"sav_train", train_root, video))
                    ann_file = video.replace(
                        ".mp4", f"_{self.annotation_set}.json")
                    with open(join(
                            self.root, f"sav_train", train_root, ann_file)
                    ) as f:
                        meta_data = json.load(f)

                    sampled_frame_count = int(np.floor(
                        meta_data["video_frame_count"] / 4
                    ))
                    self.video_names.append(meta_data["video_id"])
                    self.video_lengths.append(sampled_frame_count)
                    self.video_ids.append(meta_data["masklet_id"])

        # Load validation or test data
        if self.validation or self.test:
            # Load video name file
            video_name_file = join(
                self.root, f"sav_{self.prefix}", f"sav_{self.prefix}.txt")

            with open(video_name_file, "r") as file:
                if self.video_subset is not None:
                    for i, line in enumerate(file.read().splitlines()):
                        if i in self.video_subset:
                            self.video_names.append(line.strip())
                else:
                    self.video_names = file.read().splitlines()

            # Load video paths
            self.video_paths = [
                join(self.root, f"sav_{self.prefix}", "JPEGImages_24fps", name)
                for name in self.video_names
            ]

            # Load video lengths
            self.video_lengths = [
                int(np.floor(len(listdir(path))/4)) for path in self.video_paths
            ]

            # Load video ids
            annotation_paths = [
                join(self.root, f"sav_{self.prefix}", "Annotations_6fps", name)
                for name in self.video_names
            ]
            self.video_ids = [
                listdir(path) for path in annotation_paths
            ]

    def __load_sample__(self, idx):
        # Find video index
        video_idx = self.__find_video_idx__(idx)
        # Get frame idx
        frame_idx = self.__find_frame_idx__(idx)
        # Get video ids
        video_ids = self.video_ids[video_idx]

        # Load image and targets from training set
        if self.train:
            video_path = self.video_paths[video_idx]
            rel_video_path = video_path.split("sav_train/")[1]
            rel_video_path = rel_video_path.replace(".mp4", "")
            relative_image_name = join(
                "sav_train", rel_video_path, f"{frame_idx:05d}.jpg")
            # Load image
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx * 4)
            res, image = cap.read()
            cap.release()
            assert res, f"Could not read frame {frame_idx} from {video_path}"
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Load targets
            ann_file = video_path.replace(
                ".mp4", f"_{self.annotation_set}.json")
            with open(ann_file) as f:
                meta = json.load(f)
            last_annotated_frame = len(meta["masklet"])
            targets = None
            if frame_idx < last_annotated_frame:
                targets = dict()
                for i, rle in zip(meta["masklet_id"], meta["masklet"][round(frame_idx)]):
                    if coco_mask.area(rle) > 0:
                        mask = coco_mask.decode(rle)
                        targets[int(i)] = mask
        # Load image and targets from validation or test set
        elif self.validation or self.test:
            frame_idx = frame_idx * 4
            # Load image
            relative_image_name = join(
                f"sav_{self.prefix}", "JPEGImages_24fps",
                self.video_names[video_idx], f"{frame_idx:05d}.jpg"
            )
            image_path = join(self.root, relative_image_name)
            image = Image.open(image_path)
            image = np.array(image.convert("RGB"))
            # Load targets
            targets = None
            target_root_path = join(
                self.root, f"sav_{self.prefix}", "Annotations_6fps",
                self.video_names[video_idx],
            )
            for i in video_ids:
                target_path = join(target_root_path, i, f"{frame_idx:05d}.png")
                if exists(target_path):
                    target = Image.open(target_path)
                    target = np.array(target)
                    if targets is None:
                        targets = dict()
                    targets[int(i)] = target
        # Should not be possible
        else:
            raise NotImplementedError

        if targets is None:
            targets = dict()

        return image, targets, relative_image_name


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser(
        description="SA-V Image Dataset"
    )
    PARSER.add_argument(
        "--root", required=True, help="Path to the root of the dataset"
    )

    ARGS = PARSER.parse_args()

    dataset = SAVImageDataset(
        ARGS.root,
        split="val",
        train_subsets=[0],
        annotation_set="manual",
        sample=None
    )
    print(len(dataset))
    image, targets, points, image_name = dataset[0]
    print(image.shape, targets.keys(), image_name)
