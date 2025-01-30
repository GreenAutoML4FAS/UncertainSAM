import numpy as np

import albumentations as A


""" Template for augmentations """


class Augmentation:
    def __init__(self, name, **kwargs):
        self.name = name
        self.setting = kwargs
        self.is_invertible = False

    def __str__(self):
        title = f"{self.name}"
        for k, v in self.setting.items():
            title += f"_{k}_{v}"
        return title

    def __call__(self, image=None, points=None, masks=None, backward=False):
        # Forward transformation
        if not backward:
            # Apply transform to image (H, W, 3)
            image = self.forward_image(image) if image is not None else None
            # Apply transform to points nd.array with (n, 2)
            if self.is_invertible:
                points = self.forward_points(points) if points is not None else None
            # Apply transform to mask (n, H, W)
            if self.is_invertible:
                masks = self.forward_mask(masks) if masks is not None else None
            # Exception
            else:
                raise NotImplementedError
        # Backward transformation
        # Apply transform to image
        image = self.forward_image(image) if image is not None else None
        # Apply transform to points nd.array with
        if self.is_invertible:
            points = self.forward_points(points) if points is not None else None
        # Apply transform to mask
        if self.is_invertible:
            masks = self.forward_mask(masks) if masks is not None else None
        # Exception
        else:
            raise NotImplementedError

        return image, points, masks

    def forward_image(self, instance):
        raise NotImplementedError

    def forward_points(self, instance):
        raise NotImplementedError

    def forward_mask(self, instance):
        raise NotImplementedError

    def backward_image(self, instance):
        raise NotImplementedError

    def backward_points(self, instance):
        raise NotImplementedError

    def backward_mask(self, instance):
        raise NotImplementedError


""" Horizontal Flip augmentation """


class HorizontalFlip(Augmentation):

    def __init__(self):
        super().__init__(name="HorizontalFlip")
        self.is_invertible = True
        self.transform = A.Compose(
            [A.HorizontalFlip(p=1)],
            keypoint_params=A.KeypointParams(format='xy')
        )

    def __call__(self, image=None, points=None, masks=None, backward=False):
        # Forward transformation equals backward transformation
        # Apply transform to image and points
        if image is not None:
            if points is not None:
                t = self.transform(image=image, keypoints=points)
                points = np.asarray(t['keypoints'])
            else:
                t = self.transform(image=image)
            image = t['image']
        # Apply transform to mask
        if masks is not None:
            if masks.ndim == 2:
                masks = masks[None, :, :]
            masks = np.transpose(masks, (1, 2, 0))
            t = self.transform(image=masks)
            masks = t['image']
            masks = np.transpose(masks, (2, 0, 1))
        return image, points, masks


""" Vertical Flip augmentation """


class VerticalFlip(Augmentation):

    def __init__(self):
        super().__init__(name="VerticalFlip")
        self.is_invertible = True
        self.transform = A.Compose(
            [A.VerticalFlip(p=1)],
            keypoint_params=A.KeypointParams(format='xy')
        )

    def __call__(self, image=None, points=None, masks=None, backward=False):
        # Forward transformation equals backward transformation
        # Apply transform to image and points
        if image is not None:
            if points is not None:
                t = self.transform(image=image, keypoints=points)
                points = np.asarray(t['keypoints'])
            else:
                t = self.transform(image=image)
            image = t['image']
        # Apply transform to mask
        if masks is not None:
            if masks.ndim == 2:
                masks = masks[None, :, :]
            masks = np.transpose(masks, (1, 2, 0))
            t = self.transform(image=masks)
            masks = t['image']
            masks = np.transpose(masks, (2, 0, 1))
        return image, points, masks


""" Rotations """


class RotateClockwise(Augmentation):

    def __init__(self):
        super().__init__(name="RotateClockwise")
        self.is_invertible = True
        self.transform = A.Compose(
            [A.Affine(rotate=[90, 90], p=1, fit_output=True)],
            keypoint_params=A.KeypointParams(format='xy')
        )
        self.inverse_transform = A.Compose(
            [A.Affine(rotate=[-90, -90], p=1, fit_output=True)],
            keypoint_params=A.KeypointParams(format='xy')
        )

    def __call__(self, image=None, points=None, masks=None, backward=False):
        # Forward transformation equals backward transformation
        transform = self.transform if not backward else self.inverse_transform
        # Apply transform to image and points
        if image is not None:
            if points is not None:
                t = transform(image=image, keypoints=points)
                points = np.asarray(t['keypoints'])
            else:
                t = transform(image=image)
            image = t['image']
        # Apply transform to mask
        if masks is not None:
            if masks.ndim == 2:
                masks = masks[None, :, :]
            masks = np.transpose(masks, (1, 2, 0))
            t = transform(image=masks)
            masks = t['image']
            masks = np.transpose(masks, (2, 0, 1))
        return image, points, masks


class RotateCounterClockwise(RotateClockwise):

    def __init__(self):
        super().__init__()
        self.name = "RotateCounterClockwise"

    def __call__(self, image=None, points=None, masks=None, backward=False):
        backward = not backward
        return super().__call__(image, points, masks, backward)


""" Scale """


class Scale(Augmentation):

    def __init__(self):
        super().__init__(name="Scale")
        self.is_invertible = True
        self.transform = A.Compose(
            [A.Affine(scale=2, p=1, fit_output=True)],
            keypoint_params=A.KeypointParams(format='xy')
        )
        self.inverse_transform = A.Compose(
            [A.Affine(scale=0.5, p=1, fit_output=True)],
            keypoint_params=A.KeypointParams(format='xy')
        )

    def __call__(self, image=None, points=None, masks=None, backward=False):
        # Forward transformation equals backward transformation
        transform = self.transform if not backward else self.inverse_transform
        # Apply transform to image and points
        if image is not None:
            if points is not None:
                t = transform(image=image, keypoints=points)
                points = np.asarray(t['keypoints'])
            else:
                t = transform(image=image)
            image = t['image']
        # Apply transform to mask
        if masks is not None:
            if masks.ndim == 2:
                masks = masks[None, :, :]
            masks = np.transpose(masks, (1, 2, 0))
            t = transform(image=masks)
            masks = t['image']
            masks = np.transpose(masks, (2, 0, 1))
        return image, points, masks


""" Gaussian Blur """


class GaussianBlur(Augmentation):
    settings_min = 5
    settings_max = 45
    settings_step = 20

    def __init__(self, size):
        super().__init__(name="GaussianBlur", size=size)
        self.transform = A.GaussianBlur(blur_limit=(size, size), p=1)

    def __call__(self, image=None, points=None, masks=None, backward=False):
        # Apply transform to image and points
        if image is not None and not backward:
            image = self.transform(image=image)['image']

        return image, points, masks


""" Gaussian Noise """


class GaussNoise(Augmentation):
    settings_min = 5000
    settings_max = 15000
    settings_step = 5000

    def __init__(self, var_limit):
        super().__init__(name="GaussNoise", var_limit=var_limit)

        self.transform = A.GaussNoise(var_limit=(var_limit, var_limit), p=1)

    def __call__(self, image=None, points=None, masks=None, backward=False):
        # Apply transform to image and points
        if image is not None and not backward:
            image = self.transform(image=image)['image']

        return image, points, masks


""" Brightness """


class Brightness(Augmentation):
    settings_min = -0.75
    settings_max = 0.75
    settings_step = 0.25

    def __init__(self, limit):
        super().__init__(name="Brightness", limit=limit)
        self.transform = A.RandomBrightnessContrast(
            contrast_limit=(0, 0), brightness_limit=(limit, limit), p=1)

    def __call__(self, image=None, points=None, masks=None, backward=False):
        # Apply transform to image and points
        if image is not None and not backward:
            image = self.transform(image=image)['image']

        return image, points, masks


""" Equalize Histogramm """


class Equalize(Augmentation):

    def __init__(self):
        super().__init__(name="Equalize")
        self.transform = A.Equalize(p=1)

    def __call__(self, image=None, points=None, masks=None, backward=False):
        # Apply transform to image and points
        if image is not None and not backward:
            image = self.transform(image=image)['image']

        return image, points, masks


""" Gamma Correction """


class GammaCorrection(Augmentation):
    settings_min = 10
    settings_max = 300
    settings_step = 30
    settings_schedule = [5, 10, 50, 200, 300, 400, 500]

    def __init__(self, gamma):
        super().__init__(name="GammaCorrection", gamma=gamma)
        self.transform = A.RandomGamma(gamma_limit=(gamma, gamma), p=1)

    def __call__(self, image=None, points=None, masks=None, backward=False):
        # Apply transform to image and points
        if image is not None and not backward:
            image = self.transform(image=image)['image']

        return image, points, masks


""" ChromaticAberration """


class ChromaticAberration(Augmentation):
    settings_min = 0
    settings_max = 2
    settings_step = 0.2
    settings_schedule = [0.05, 0.1, 0.3, 0.5]

    def __init__(self, distortion):
        super().__init__(name="ChromaticAberration", limit=distortion)
        self.transform = A.ChromaticAberration(
            primary_distortion_limit=(distortion, distortion),
            secondary_distortion_limit=(
                distortion * 2.5, distortion * 2.5), p=1)

    def __call__(self, image=None, points=None, masks=None, backward=False):
        # Apply transform to image and points
        if image is not None and not backward:
            image = self.transform(image=image)['image']

        return image, points, masks


""" ImageCompression """


class ImageCompression(Augmentation):
    settings_min = 1
    settings_max = 100
    settings_step = 10
    settings_schedule = [1, 5, 10, 30]

    def __init__(self, quality):
        super().__init__(name="ImageCompression", quality=quality)
        self.transform = A.ImageCompression(
            quality_range=(quality, quality), p=1)

    def __call__(self, image=None, points=None, masks=None, backward=False):
        # Apply transform to image and points
        if image is not None and not backward:
            image = self.transform(image=image)['image']

        return image, points, masks


""" Gray """


class Gray(Augmentation):

    def __init__(self):
        super().__init__(name="Gray")
        self.transform = A.ToGray(p=1)

    def __call__(self, image=None, points=None, masks=None, backward=False):
        # Apply transform to image and points
        if image is not None and not backward:
            image = self.transform(image=image)['image']

        return image, points, masks


""" CLAHE """


class CLAHE(Augmentation):
    settings_min = 1
    settings_max = 11
    settings_step = 2
    def __init__(self, tile_grid_size):
        super().__init__(name="CLAHE", tile_grid_size=tile_grid_size)
        limit = tile_grid_size
        tile_grid_size = 16
        self.transform = A.CLAHE(clip_limit=(limit, limit),
            tile_grid_size=(tile_grid_size, tile_grid_size), p=1)

    def __call__(self, image=None, points=None, masks=None, backward=False):
        # Apply transform to image and points
        if image is not None and not backward:
            image = self.transform(image=image)['image']

        return image, points, masks

