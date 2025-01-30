import numpy as np
import gc
import pickle
import datetime
from os.path import join, dirname, exists, basename
from os import makedirs, listdir
import torch


class TrainingContainer:
    """
    A data container that stores data for training in an efficient way
    """
    def __init__(self):
        self.iou_tokens = list()
        self.mask_tokens = list()
        self.metrics = list()
        self.image_names = list()
        self.model = list()
        self.is_offloaded = False
        self.subset = None

    def split(self, split: str):
        """
        Split into train and validation set
        :param split: train or val
        :return:
        """
        if split == "train":
            _is = [i < 0.8 * self.__len__() for i in range(self.__len__())]
        elif split == "val":
            _is = [i >= 0.8 * self.__len__() for i in range(self.__len__())]
        else:
            raise NotImplementedError

        if not self.is_offloaded:
            self.iou_tokens = [
                x for v, x in zip(_is, self.iou_tokens) if v is True]
            self.mask_tokens = [
                x for v, x in zip(_is, self.mask_tokens) if v is True]
            self.metrics = [
                x for v, x in zip(_is, self.metrics) if v is True]
            self.image_names = [
                x for v, x in zip(_is, self.image_names) if v is True]
            self.model = [
                x for v, x in zip(_is, self.model) if v is True]

            self.iou_tokens = torch.stack(self.iou_tokens)
            self.mask_tokens = torch.stack(self.mask_tokens)
        else:
            self.subset = np.asarray([i for i, v in enumerate(_is) if v])

    def append(
            self,
            iou_token,
            mask_token,
            metrics,
            image_name,
            model
    ):
        assert not self.is_offloaded, "Cannot append to offloaded container"
        self.iou_tokens.append(iou_token)
        self.mask_tokens.append(mask_token)
        self.metrics.append(metrics)
        self.image_names.append(image_name)
        self.model.append(model)

    def __len__(self):
        if not self.is_offloaded:
            return len(self.iou_tokens)
        else:
            return len(self.subset)

    def filter(
            self,
            model: str = None,
            no_augmented: bool = False,
    ):
        """
        Removes all elements that do not match the filter criteria.
        :param model: One of large, base+, tiny, and small
        :param no_augmented: if true, only non-augmented data is returned
        :return:
        """
        if not self.is_offloaded:
            _is = [True for _ in range(self.__len__())]
        else:
            _is = np.zeros(len(self.iou_tokens), dtype=bool)
            _is[self.subset] = True

        if model is not None:
            _is = [m == model and v for v, m in zip(_is, self.model)]

        if no_augmented:
            if not self.is_offloaded:
                _is = [
                    not m["is_augmented"] and v for v, m in zip(_is, self.metrics)
                ]
            else:
                _is = [
                    not m and v for v, m in zip(_is, self.metrics["is_augmented"])
                ]

        if not self.is_offloaded:
            self.iou_tokens = [
                x for v, x in zip(_is, self.iou_tokens) if v is True]
            self.mask_tokens = [
                x for v, x in zip(_is, self.mask_tokens) if v is True]
            self.metrics = [
                x for v, x in zip(_is, self.metrics) if v is True]
            self.image_names = [
                x for v, x in zip(_is, self.image_names) if v is True]
            self.model = [
                x for v, x in zip(_is, self.model) if v is True]

            self.iou_tokens = torch.stack(self.iou_tokens)
            self.mask_tokens = torch.stack(self.mask_tokens)
        else:
            self.subset = np.asarray([i for i, v in enumerate(_is) if v])


def load(filename) -> TrainingContainer:
    """
    Loads a training container from a file
    """
    before = datetime.datetime.now()
    gc.disable()
    if filename.endswith(".pth"):
        ret = torch.load(filename)
    else:
        raise NotImplementedError
    gc.enable()
    after = datetime.datetime.now()

    # Check if the file is offloaded
    if ret.is_offloaded:
        storage_name = join(dirname(filename), "STORAGE_" + basename(filename))
        ret.iou_tokens = np.load(join(storage_name, "iou_tokens.npy"), mmap_mode="r")
        ret.mask_tokens = np.load(join(storage_name, "mask_tokens.npy"), mmap_mode="r")
        ret.image_names = open(join(storage_name, "image_names.txt"), "r").read().split("\n")
        ret.model = open(join(storage_name, "model.txt"), "r").read().split("\n")
        ret.metrics = dict()
        metrics = list()
        for metric in listdir(storage_name):
            if metric.startswith("metric_") and metric.endswith(".npy"):
                metrics.append(metric.replace("metric_", "").replace(".npy", ""))
        for key in metrics:
            try:
                ret.metrics[key] = np.load(join(storage_name, f"metric_{key}.npy"), mmap_mode="r")
            except FileNotFoundError:
                print("File not found:", join(storage_name, f"metric_{key}.npy"))
        ret.subset = np.arange(len(ret.iou_tokens))

    print(f"Loading {filename} took {after - before}")
    return ret


def save(filename, container, offload=False):
    """
    Saves a training container to a file
    :param filename:
    :param container:
    :param offload:  If true, the data instances are stored on disk otherwise in memory
    :return:
    """
    container.iou_tokens = torch.stack(container.iou_tokens)
    container.mask_tokens = torch.stack(container.mask_tokens)

    if offload:
        storage_name = join(dirname(filename), "STORAGE_" + basename(filename))
        makedirs(storage_name, exist_ok=True)
        np.save(join(storage_name, "iou_tokens.npy"), container.iou_tokens.numpy())
        np.save(join(storage_name, "mask_tokens.npy"), container.mask_tokens.numpy())

        keys = list(container.metrics[0].keys())
        for key in keys:
            np.save(
                join(storage_name, f"metric_{key}.npy"),
                np.asarray([m[key] for m in container.metrics]),
            )

        with open(join(storage_name, "image_names.txt"), "w+") as file:
            file.write("\n".join(container.image_names))
        with open(join(storage_name, "model.txt"), "w+") as file:
            file.write("\n".join(container.model))

        container.is_offloaded = True
        container.iou_tokens = None
        container.mask_tokens = None
        container.metrics = None
        container.model = None
        container.image_names = None

    path = dirname(filename)
    if not exists(path):
        makedirs(path, exist_ok=True)
    if filename.endswith(".pkl"):
        with open(filename, "wb+") as file:
            pickle.dump(container, file, protocol=pickle.HIGHEST_PROTOCOL)
    if filename.endswith(".pth"):
        torch.save(container, filename)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    assert os.path.join(args.file)

    container = load(args.file)
    print(container.__len__())
    print(len(container))
