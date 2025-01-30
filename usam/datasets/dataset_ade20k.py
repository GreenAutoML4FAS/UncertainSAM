from os import listdir
from os.path import join
from PIL import Image
import numpy as np

from usam.datasets.dataset import ImageDataset


class ADE20KImageDataset(ImageDataset):
    def __init__(
            self,
            root,
            **kwargs,
    ):

        super(ADE20KImageDataset, self).__init__(root, **kwargs)
        assert not self.test, "Test split not supported"

    def __parse__(self):
        # Load video name file
        if self.train:
            self.set = "training"
        elif self.validation:
            self.set = "validation"
        else:
            raise ValueError("Invalid split")

        self.video_names = listdir(join(self.root, "images", self.set))
        self.video_names.sort()
        # Load video paths
        self.video_paths = [
            join(self.root, "images", self.set, name)
            for name in self.video_names
        ]

        # Load video lengths
        self.video_lengths = [1 for _ in self.video_paths]

    def __load_sample__(self, idx):
        # Find video index
        video_idx = self.__find_video_idx__(idx)
        # Load image
        relative_image_name = join("images", self.set, self.video_names[video_idx])
        image_path = join(self.root, relative_image_name)
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))

        target_path = join(self.root, "annotations", self.set, self.video_names[video_idx])
        target_path = target_path.replace(".jpg", ".png")
        target = np.array(Image.open(target_path))
        targets = dict()
        for i in np.unique(target):
            if i != 0:
                targets[int(i)] = target == i

        return image, targets, relative_image_name


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser(
        description="Cityscapes Image Dataset"
    )
    PARSER.add_argument(
        "--root", required=True, help="Path to the root of the dataset"
    )

    ARGS = PARSER.parse_args()

    dataset = ADE20KImageDataset(ARGS.root, split="val")
    print(len(dataset))
    image, targets, centers, sample_points, image_name = dataset[0]
    print(image.shape, targets.keys(), image_name)

