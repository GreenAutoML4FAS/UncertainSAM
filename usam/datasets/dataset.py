import numpy as np
from torch.utils.data import Dataset
import cv2


class ImageDataset(Dataset):
    """
    Base class for image datasets. This class provides the basic functionality
    for loading images from a dataset. It should be subclassed by the specific
    dataset implementation. The images are parsed independently of the video
    """

    def __init__(
            self,
            root: str,
            split: str = "train",
            video_subset: list = None,
            augmentations=None,
            start_frame=None,
            end_frame=None,
            sample: int = None,
    ):
        """
        Initializes the dataset. The root directory of the dataset must be
        provided. The dataset can be split into a train, validation and test
        set. The video_subset parameter can be used to only load a subset of
        the videos.

        :param root: The root directory of the dataset
        :param train: if True, the train set is loaded
        :param validation: if True, the validation set is loaded
        :param test: if True, the test set is loaded
        :param video_subset: A list of names of the videos that should be loaded
        """
        super(Dataset, self).__init__()
        self.root = root
        self.video_subset = video_subset
        self.augmentations = augmentations
        self.sample = sample

        assert split in ["train", "val", "test"], \
            "Split must be train, val or test"

        self.train = True if split == "train" else False
        self.validation = True if split == "val" else False
        self.test = True if split == "test" else False

        self.video_names = []
        self.video_paths = []
        self.video_lengths = []

        self.__parse__()

        length = sum(self.video_lengths)
        self.start = 0 if start_frame is None else start_frame
        self.end = length if end_frame is None else min(end_frame, length)
        self.length = self.end - self.start

        self.video_lengths_cumsum = np.cumsum([0] + self.video_lengths)

    def __parse__(self):
        """ Parses the dataset and loads the video names, paths and lengths """
        raise NotImplementedError

    def __len__(self):
        """ Returns the number of images in the dataset """
        if self.sample is None:
            return self.length
        else:
            return int(np.floor(self.length / self.sample))

    def __load_sample__(self, idx):
        """
        Loads a sample from the dataset. This method should be implemented by
        the subclass.

        :param idx: Index of the image that should be returned
        :return: np.ndarray, {id: rle}, str
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset. This method is called by the
        PyTorch DataLoader.

        Internally it just calls __load_sample__ and returns the result.

        :param idx: Index of the image that should be returned
        :return: np.ndarray, {id: rle}, str
        """
        idx += self.start

        if self.sample is not None:
            idx = idx * self.sample

        image, targets, image_name = self.__load_sample__(idx)
        # Check if targets have an non zero element in its mask
        if targets is not None:
            targets = {k: v for k, v in targets.items() if v.sum() > 0}
            if len(targets) == 0:
                targets = None

        # Apply augmentations
        centers, sample_points = [], []
        for i, mask in targets.items() if targets is not None else []:
            # Erode mask by 2 pixels
            mask = (mask * 255).astype(np.uint8)
            eroded_mask = cv2.erode(
                mask, np.ones((5, 5), np.uint8), iterations=1)
            mask = (mask / 255).astype(bool)
            eroded_mask = (eroded_mask / 255).astype(bool)
            if np.unique(eroded_mask).size > 1:
                _mask = eroded_mask.astype(bool)
            else:
                _mask = mask
            # Calculate point of the masks that is closest to its center
            foreground = np.argwhere(_mask == 1)
            num_fg = len(foreground)
            aleatoric_samples = 7
            sample_inds = np.random.choice(
                np.arange(num_fg), aleatoric_samples,
                replace=False if num_fg >= aleatoric_samples else True
            )
            sample_points.append([foreground[i][::-1] for i in sample_inds])
            center = foreground.mean(axis=0)
            distance = np.linalg.norm(foreground - center, axis=1)
            closest_point = np.argmin(distance)
            closest_point = foreground[closest_point][::-1]
            centers.append(closest_point)
        centers = np.array(centers)
        sample_points = [np.array(x) for x in sample_points]
        if self.augmentations is None:
            return image, targets, centers, sample_points, image_name

        # Iterate over the augmentations and apply them
        augmented_images = []
        augmented_points = []
        augmented_smpls = []
        augmentations = []
        if targets is not None:
            for aug in self.augmentations:
                augmented_image, augmented_pts, _ = (
                    aug(
                        image=image,
                        points=np.concatenate([centers] + sample_points)
                    ))
                augmented_masses = augmented_pts[0:len(centers)]
                augmented_smpl = [
                    augmented_pts[
                        len(centers)+aleatoric_samples*i:len(centers)+aleatoric_samples*(i+1)]
                    for i in range(len(centers))
                ]
                augmented_images.append(augmented_image)
                augmented_points.append(augmented_masses)
                augmented_smpls.append(augmented_smpl)
                augmentations.append(aug)

        return (image, targets, centers, sample_points, image_name,
                augmented_images, augmented_points, augmented_smpls, augmentations)

    def __find_video_idx__(self, idx):
        """ Returns the index of the video that contains the frame index """
        return np.argmax(self.video_lengths_cumsum > idx) - 1

    def __find_frame_idx__(self, idx):
        """ Returns the relative frame index within the video """
        video_idx = self.__find_video_idx__(idx)
        if video_idx == 0:
            frame_idx = idx
        else:
            frame_idx = idx - self.video_lengths_cumsum[video_idx]
        return frame_idx


