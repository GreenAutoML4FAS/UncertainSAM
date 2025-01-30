import numpy as np
from torch.utils.data import Dataset
import torch as t
from os.path import exists

from usam.datasets.training_container import load


class ContainerDataset(Dataset):
    def __init__(
            self,
            filename: str,
            metric: str,
            split: str = "trainval"
    ):
        assert exists(filename), f"{filename} does not exist!"
        assert split in ["trainval", "val", "train"]

        self.container = load(filename)
        self.metric = metric
        self.split = split
        self.prepared_metrics = None

        if split != "trainval":
            self.container.split(split)

    def __len__(self):
        return len(self.container)

    def __getitem__(self, idx):

        if not self.container.is_offloaded:
            target = self.container.metrics[idx][self.metric]
            iou_token = self.container.iou_tokens[idx]
            mask_token = self.container.mask_tokens[idx]
        else:
            iou_token = self.container.iou_tokens[self.container.subset[idx]]
            mask_token = self.container.mask_tokens[self.container.subset[idx]]
            target = self.container.metrics[self.metric][self.container.subset[idx]]
            # Load mmap array to cpus
            target = np.array(target)
            iou_token = np.array(iou_token)
            mask_token = np.array(mask_token)
            iou_token = t.from_numpy(iou_token)
            mask_token = t.from_numpy(mask_token)
            target = t.from_numpy(target)

        if self.metric in [
            "prompt_gap", "task_gap", "model_gap",
        ]:
            target = 0.5 + target

        target = np.clip(target, 0, 1)

        return t.cat([iou_token, mask_token]), target


