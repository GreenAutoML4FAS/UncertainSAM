from usam.training.augmentations import (
    GaussianBlur, GaussNoise, ImageCompression
)


class SimpleAugmentationScheduler:
    def __init__(self):
        self.augmentations = list()

        self.augmentations.append(ImageCompression(10))
        self.augmentations.append(ImageCompression(30))
        self.augmentations.append(GaussNoise(5000))
        self.augmentations.append(GaussianBlur(5))

        self.num = 0
        self.end = len(self.augmentations)

    def __iter__(self):
        self.num = 0
        return self

    def __next__(self):
        if self.num >= self.end:
            raise StopIteration
        else:
            self.num += 1
            return self.augmentations[self.num - 1]

    def __len__(self):
        return len(self.augmentations)


if __name__ == "__main__":
    augmentations = SimpleAugmentationScheduler()
    for i, aug in enumerate(augmentations):
        print(f"{i}: {aug.name} | {aug.setting}")
