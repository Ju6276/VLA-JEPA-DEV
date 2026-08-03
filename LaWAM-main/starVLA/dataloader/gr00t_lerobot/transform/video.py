# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, ClassVar, Literal

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as T
from einops import rearrange
from pydantic import Field, PrivateAttr, field_validator
from PIL import Image

from ..schema import DatasetMetadata
from .base import ModalityTransform


class VideoTransform(ModalityTransform):
    # Configurable attributes
    backend: str = Field(
        default="torchvision", description="The backend to use for the transformations"
    )

    # Model variables
    _train_transform: Callable | None = PrivateAttr(default=None)
    _eval_transform: Callable | None = PrivateAttr(default=None)
    _original_resolutions: dict[str, tuple[int, int]] = PrivateAttr(default_factory=dict)

    # Model constants
    _INTERPOLATION_MAP: ClassVar[dict[str, dict[str, Any]]] = PrivateAttr(
        {
            "nearest": {
                "albumentations": cv2.INTER_NEAREST,
                "torchvision": T.InterpolationMode.NEAREST,
            },
            "linear": {
                "albumentations": cv2.INTER_LINEAR,
                "torchvision": T.InterpolationMode.BILINEAR,
            },
            "cubic": {
                "albumentations": cv2.INTER_CUBIC,
                "torchvision": T.InterpolationMode.BICUBIC,
            },
            "area": {
                "albumentations": cv2.INTER_AREA,
                "torchvision": None,  # Torchvision does not support this interpolation mode
            },
            "lanczos4": {
                "albumentations": cv2.INTER_LANCZOS4,  # Lanczos with a 4x4 filter
                "torchvision": T.InterpolationMode.LANCZOS,  # Torchvision does not specify filter size, might be different from 4x4
            },
            "linear_exact": {
                "albumentations": cv2.INTER_LINEAR_EXACT,
                "torchvision": None,  # Torchvision does not support this interpolation mode
            },
            "nearest_exact": {
                "albumentations": cv2.INTER_NEAREST_EXACT,
                "torchvision": T.InterpolationMode.NEAREST_EXACT,
            },
            "max": {
                "albumentations": cv2.INTER_MAX,
                "torchvision": None,
            },
        }
    )

    @property
    def train_transform(self) -> Callable:
        assert (
            self._train_transform is not None
        ), "Transform is not set. Please call set_metadata() before calling apply()."
        return self._train_transform

    @train_transform.setter
    def train_transform(self, value: Callable):
        self._train_transform = value

    @property
    def eval_transform(self) -> Callable | None:
        return self._eval_transform

    @eval_transform.setter
    def eval_transform(self, value: Callable | None):
        self._eval_transform = value

    @property
    def original_resolutions(self) -> dict[str, tuple[int, int]]:
        assert (
            self._original_resolutions is not None
        ), "Original resolutions are not set. Please call set_metadata() before calling apply()."
        return self._original_resolutions

    @original_resolutions.setter
    def original_resolutions(self, value: dict[str, tuple[int, int]]):
        self._original_resolutions = value

    def check_input(self, data: dict[str, Any]):
        if self.backend == "torchvision":
            for key in self.apply_to:
                assert isinstance(data[key], torch.Tensor), f"Video {key} is not a torch tensor"
                assert data[key].ndim in [
                    4,
                    5,
                ], f"Expected video {key} to have 4 or 5 dimensions (T, C, H, W or T, B, C, H, W), got {data[key].ndim}"
        elif self.backend == "albumentations":
            for key in self.apply_to:
                assert isinstance(data[key], np.ndarray), f"Video {key} is not a numpy array"
                assert data[key].ndim in [
                    4,
                    5,
                ], f"Expected video {key} to have 4 or 5 dimensions (T, C, H, W or T, B, C, H, W), got {data[key].ndim}"
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        super().set_metadata(dataset_metadata)
        self.original_resolutions = {}
        for key in self.apply_to:
            split_keys = key.split(".")
            assert len(split_keys) == 2, f"Invalid key: {key}. Expected format: modality.key"
            sub_key = split_keys[1]
            if sub_key in dataset_metadata.modalities.video:
                self.original_resolutions[key] = dataset_metadata.modalities.video[
                    sub_key
                ].resolution
            else:
                raise ValueError(
                    f"Video key {sub_key} not found in dataset metadata. Available keys: {dataset_metadata.modalities.video.keys()}"
                )
        train_transform = self.get_transform(mode="train")
        eval_transform = self.get_transform(mode="eval")
        if self.backend == "albumentations":
            self.train_transform = A.ReplayCompose(transforms=[train_transform])  # type: ignore
            if eval_transform is not None:
                self.eval_transform = A.ReplayCompose(transforms=[eval_transform])  # type: ignore
        else:
            assert train_transform is not None, "Train transform must be set"
            self.train_transform = train_transform
            self.eval_transform = eval_transform

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.training:
            transform = self.train_transform
        else:
            transform = self.eval_transform
            if transform is None:
                return data
        assert (
            transform is not None
        ), "Transform is not set. Please call set_metadata() before calling apply()."
        try:
            self.check_input(data)
        except AssertionError as e:
            raise ValueError(
                f"Input data does not match the expected format for {self.__class__.__name__}: {e}"
            ) from e

        # Concatenate views while preserving each view's own temporal length.
        views = [data[key] for key in self.apply_to]
        is_batched = views[0].ndim == 5
        bs = int(views[0].shape[0]) if is_batched else 1
        view_lengths: list[int] = []
        flat_views: list[torch.Tensor | np.ndarray] = []

        first_view = views[0]
        if isinstance(first_view, torch.Tensor):
            for idx, view in enumerate(views):
                if not isinstance(view, torch.Tensor):
                    raise ValueError(f"Mixed view types are not supported, got {type(view)} at index {idx}.")
                if is_batched:
                    if view.ndim != 5:
                        raise ValueError(f"Expected batched video tensor with 5 dims, got {tuple(view.shape)}.")
                    if int(view.shape[0]) != bs:
                        raise ValueError(
                            f"All batched views must share the same batch size, got {int(view.shape[0])} and {bs}."
                        )
                    view_lengths.append(int(view.shape[1]))
                    flat_views.append(rearrange(view, "b t c h w -> (b t) c h w"))
                else:
                    if view.ndim != 4:
                        raise ValueError(f"Expected video tensor with 4 dims, got {tuple(view.shape)}.")
                    view_lengths.append(int(view.shape[0]))
                    flat_views.append(view)
            views = torch.cat(flat_views, dim=0)
        elif isinstance(first_view, np.ndarray):
            for idx, view in enumerate(views):
                if not isinstance(view, np.ndarray):
                    raise ValueError(f"Mixed view types are not supported, got {type(view)} at index {idx}.")
                if is_batched:
                    if view.ndim != 5:
                        raise ValueError(f"Expected batched video array with 5 dims, got {view.shape}.")
                    if int(view.shape[0]) != bs:
                        raise ValueError(
                            f"All batched views must share the same batch size, got {int(view.shape[0])} and {bs}."
                        )
                    view_lengths.append(int(view.shape[1]))
                    flat_views.append(rearrange(view, "b t c h w -> (b t) c h w"))
                else:
                    if view.ndim != 4:
                        raise ValueError(f"Expected video array with 4 dims, got {view.shape}.")
                    view_lengths.append(int(view.shape[0]))
                    flat_views.append(view)
            views = np.concatenate(flat_views, axis=0)
        else:
            raise ValueError(f"Unsupported view type: {type(first_view)}")

        # Apply the transform
        if self.backend == "torchvision":
            views = transform(views)
        elif self.backend == "albumentations":
            assert isinstance(transform, A.ReplayCompose), "Transform must be a ReplayCompose"
            first_frame = views[0]
            transformed = transform(image=first_frame)
            replay_data = transformed["replay"]
            transformed_first_frame = transformed["image"]

            if len(views) > 1:
                # Apply the same transformations to the rest of the frames
                transformed_frames = [
                    transform.replay(replay_data, image=frame)["image"] for frame in views[1:]
                ]
                # Add the first frame back
                transformed_frames = [transformed_first_frame] + transformed_frames
            else:
                # If there is only one frame, just make a list with one frame
                transformed_frames = [transformed_first_frame]

            # Delete the replay data to save memory
            del replay_data
            views = np.stack(transformed_frames, 0)

        else:
            raise ValueError(f"Backend {self.backend} not supported")

        # Split views according to each view's original temporal length.
        start = 0
        for key, view_len in zip(self.apply_to, view_lengths):
            chunk_len = int(view_len * bs) if is_batched else int(view_len)
            end = start + chunk_len
            chunk = views[start:end]
            if is_batched:
                chunk = rearrange(chunk, "(b t) c h w -> b t c h w", b=bs, t=view_len)
            data[key] = chunk
            start = end
        return data

    @classmethod
    def _validate_interpolation(cls, interpolation: str):
        if interpolation not in cls._INTERPOLATION_MAP:
            raise ValueError(f"Interpolation mode {interpolation} not supported")

    def _get_interpolation(self, interpolation: str, backend: str = "torchvision"):
        """
        Get the interpolation mode for the given backend.

        Args:
            interpolation (str): The interpolation mode.
            backend (str): The backend to use.

        Returns:
            Any: The interpolation mode for the given backend.
        """
        return self._INTERPOLATION_MAP[interpolation][backend]

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        raise NotImplementedError(
            "set_transform is not implemented for VideoTransform. Please implement this function to set the transforms."
        )


class VideoCrop(VideoTransform):
    height: int | None = Field(default=None, description="The height of the input image")
    width: int | None = Field(default=None, description="The width of the input image")
    scale: float = Field(
        ...,
        description="The scale of the crop. The crop size is (width * scale, height * scale)",
    )

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Get the transform for the given mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable: If mode is "train", return a random crop transform. If mode is "eval", return a center crop transform.
        """
        # 1. Check the input resolution
        assert (
            len(set(self.original_resolutions.values())) == 1
        ), f"All video keys must have the same resolution, got: {self.original_resolutions}"
        if self.height is None:
            assert self.width is None, "Height and width must be either both provided or both None"
            self.width, self.height = self.original_resolutions[self.apply_to[0]]
        else:
            assert (
                self.width is not None
            ), "Height and width must be either both provided or both None"
        # 2. Create the transform
        size = (int(self.height * self.scale), int(self.width * self.scale))
        if self.backend == "torchvision":
            if mode == "train":
                return T.RandomCrop(size)
            elif mode == "eval":
                return T.CenterCrop(size)
            else:
                raise ValueError(f"Crop mode {mode} not supported")
        elif self.backend == "albumentations":
            if mode == "train":
                return A.RandomCrop(height=size[0], width=size[1], p=1)
            elif mode == "eval":
                return A.CenterCrop(height=size[0], width=size[1], p=1)
            else:
                raise ValueError(f"Crop mode {mode} not supported")
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    def check_input(self, data: dict[str, Any]):
        super().check_input(data)
        # Check the input resolution
        for key in self.apply_to:
            if self.backend == "torchvision":
                height, width = data[key].shape[-2:]
            elif self.backend == "albumentations":
                height, width = data[key].shape[-3:-1]
            else:
                raise ValueError(f"Backend {self.backend} not supported")
            assert (
                height == self.height and width == self.width
            ), f"Video {key} has invalid shape {height, width}, expected {self.height, self.width}"


class VideoOffsetCrop(VideoTransform):
    top: int = Field(..., description="Top offset of the crop.")
    left: int = Field(..., description="Left offset of the crop.")
    height: int = Field(..., description="Crop height.")
    width: int = Field(..., description="Crop width.")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        del mode
        if self.backend != "torchvision":
            raise ValueError(f"Backend {self.backend} not supported")
        top = int(self.top)
        left = int(self.left)
        height = int(self.height)
        width = int(self.width)
        return lambda frames: frames[..., top : top + height, left : left + width]

    def check_input(self, data: dict[str, Any]):
        super().check_input(data)
        for key in self.apply_to:
            if self.backend != "torchvision":
                raise ValueError(f"Backend {self.backend} not supported")
            height, width = data[key].shape[-2:]
            assert self.top >= 0 and self.left >= 0, (
                f"VideoOffsetCrop expects non-negative offsets, got top={self.top}, left={self.left}"
            )
            assert self.height > 0 and self.width > 0, (
                f"VideoOffsetCrop expects positive crop size, got height={self.height}, width={self.width}"
            )
            assert self.top + self.height <= height and self.left + self.width <= width, (
                f"Crop window {(self.top, self.left, self.height, self.width)} exceeds input "
                f"resolution {(height, width)} for key {key}"
            )


class VideoResize(VideoTransform):
    height: int = Field(..., description="The height of the resize")
    width: int = Field(..., description="The width of the resize")
    interpolation: str = Field(default="linear", description="The interpolation mode")
    antialias: bool = Field(default=True, description="Whether to apply antialiasing")

    @field_validator("interpolation")
    def validate_interpolation(cls, v):
        cls._validate_interpolation(v)
        return v

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Get the resize transform. Same transform for both train and eval.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable: The resize transform.
        """
        interpolation = self._get_interpolation(self.interpolation, self.backend)
        if interpolation is None:
            raise ValueError(
                f"Interpolation mode {self.interpolation} not supported for torchvision"
            )
        if self.backend == "torchvision":
            size = (self.height, self.width)
            return T.Resize(size, interpolation=interpolation, antialias=self.antialias)
        elif self.backend == "albumentations":
            return A.Resize(
                height=self.height,
                width=self.width,
                interpolation=interpolation,
                p=1,
            )
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoRandomRotation(VideoTransform):
    degrees: float | tuple[float, float] = Field(
        ..., description="The degrees of the random rotation"
    )
    interpolation: str = Field("linear", description="The interpolation mode")

    @field_validator("interpolation")
    def validate_interpolation(cls, v):
        cls._validate_interpolation(v)
        return v

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the random rotation transform, only used in train mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable | None: The random rotation transform. None for eval mode.
        """
        if mode == "eval":
            return None
        interpolation = self._get_interpolation(self.interpolation, self.backend)
        if interpolation is None:
            raise ValueError(
                f"Interpolation mode {self.interpolation} not supported for torchvision"
            )
        if self.backend == "torchvision":
            return T.RandomRotation(self.degrees, interpolation=interpolation)  # type: ignore
        elif self.backend == "albumentations":
            return A.Rotate(limit=self.degrees, interpolation=interpolation, p=1)
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoHorizontalFlip(VideoTransform):
    p: float = Field(..., description="The probability of the horizontal flip")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the horizontal flip transform, only used in train mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable | None: If mode is "train", return a horizontal flip transform. If mode is "eval", return None.
        """
        if mode == "eval":
            return None
        if self.backend == "torchvision":
            return T.RandomHorizontalFlip(self.p)
        elif self.backend == "albumentations":
            return A.HorizontalFlip(p=self.p)
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoGrayscale(VideoTransform):
    p: float = Field(..., description="The probability of the grayscale transformation")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the grayscale transform, only used in train mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable | None: If mode is "train", return a grayscale transform. If mode is "eval", return None.
        """
        if mode == "eval":
            return None
        if self.backend == "torchvision":
            return T.RandomGrayscale(self.p)
        elif self.backend == "albumentations":
            return A.ToGray(p=self.p)
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoColorJitter(VideoTransform):
    brightness: float | tuple[float, float] = Field(
        ..., description="The brightness of the color jitter"
    )
    contrast: float | tuple[float, float] = Field(
        ..., description="The contrast of the color jitter"
    )
    saturation: float | tuple[float, float] = Field(
        ..., description="The saturation of the color jitter"
    )
    hue: float | tuple[float, float] = Field(..., description="The hue of the color jitter")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the color jitter transform, only used in train mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable | None: If mode is "train", return a color jitter transform. If mode is "eval", return None.
        """
        if mode == "eval":
            return None
        if self.backend == "torchvision":
            return T.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue,
            )
        elif self.backend == "albumentations":
            return A.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue,
                p=1,
            )
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoRandomGrayscale(VideoTransform):
    p: float = Field(..., description="The probability of the grayscale transformation")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the grayscale transform, only used in train mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable | None: If mode is "train", return a grayscale transform. If mode is "eval", return None.
        """
        if mode == "eval":
            return None
        if self.backend == "torchvision":
            return T.RandomGrayscale(self.p)
        elif self.backend == "albumentations":
            return A.ToGray(p=self.p)
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoRandomPosterize(VideoTransform):
    bits: int = Field(..., description="The number of bits to posterize the image")
    p: float = Field(..., description="The probability of the posterize transformation")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the posterize transform, only used in train mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable | None: If mode is "train", return a posterize transform. If mode is "eval", return None.
        """
        if mode == "eval":
            return None
        if self.backend == "torchvision":
            return T.RandomPosterize(bits=self.bits, p=self.p)
        elif self.backend == "albumentations":
            return A.Posterize(num_bits=self.bits, p=self.p)
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoToTensorUint8(VideoTransform):
    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        if self.backend == "torchvision":
            return self.__class__.to_tensor_uint8
        raise ValueError(f"Backend {self.backend} not supported")

    def check_input(self, data: dict):
        VideoToTensor.check_input(self, data)

    @staticmethod
    def to_tensor_uint8(frames: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()


class VideoMaxAspectCrop(VideoTransform):
    max_aspect: float = Field(..., description="Maximum allowed width/height aspect ratio.")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        if self.backend != "torchvision":
            raise ValueError(f"Backend {self.backend} not supported")
        max_aspect = float(self.max_aspect)
        return lambda frames: self.crop_to_max_aspect(frames, max_aspect=max_aspect)

    @staticmethod
    def crop_to_max_aspect(frames: torch.Tensor, *, max_aspect: float) -> torch.Tensor:
        if frames.ndim != 4 or int(frames.shape[1]) != 3:
            raise ValueError(f"Expected frames tensor with shape [N, 3, H, W], got {tuple(frames.shape)}.")
        height = int(frames.shape[-2])
        width = int(frames.shape[-1])
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid frame spatial shape HxW={height}x{width}.")
        if (width / height) <= float(max_aspect):
            return frames.contiguous()
        crop_width = max(1, int(np.floor(height * float(max_aspect))))
        left = (width - crop_width) // 2
        return frames[..., left : left + crop_width].contiguous()


class VideoToTensor(VideoTransform):
    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Get the to tensor transform. Same transform for both train and eval.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable: The to tensor transform.
        """
        if self.backend == "torchvision":
            return self.__class__.to_tensor
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    def check_input(self, data: dict):
        """Check if the input data has the correct shape.
        Expected video shape: [T, H, W, C], dtype np.uint8
        """
        for key in self.apply_to:
            assert key in data, f"Key {key} not found in data. Available keys: {data.keys()}"
            assert data[key].ndim in [
                4,
                5,
            ], f"Video {key} must have 4 or 5 dimensions, got {data[key].ndim}"
            assert (
                data[key].dtype == np.uint8
            ), f"Video {key} must have dtype uint8, got {data[key].dtype}"
            input_resolution = tuple(int(v) for v in data[key].shape[-3:-1][::-1])
            assert all(v > 0 for v in input_resolution), (
                f"Video {key} must have positive spatial dimensions, got {input_resolution}. "
                f"Full shape: {data[key].shape}"
            )
            channels = int(data[key].shape[-1])
            assert channels == 3, (
                f"Video {key} must have 3 channels, got {channels}. "
                f"Full shape: {data[key].shape}"
            )

    @staticmethod
    def to_tensor(frames: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor efficiently.

        Args:
            frames: numpy array of shape [T, H, W, C] in uint8 format
        Returns:
            tensor of shape [T, C, H, W] in range [0, 1]
        """
        frames_tensor = torch.from_numpy(frames).to(torch.float32) / 255.0
        return frames_tensor.permute(0, 3, 1, 2)  # [T, C, H, W]


class VideoToUint8Tensor(VideoTransform):
    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        if self.backend == "torchvision":
            return self.__class__.to_uint8_tensor
        raise ValueError(f"Backend {self.backend} not supported")

    def check_input(self, data: dict):
        super().check_input(data)
        for key in self.apply_to:
            assert torch.is_floating_point(data[key]), (
                f"Video {key} must be a floating tensor before uint8 conversion, got {data[key].dtype}"
            )
            channels = int(data[key].shape[-3])
            assert channels == 3, (
                f"Video {key} must have 3 channels in CHW layout, got shape {tuple(data[key].shape)}"
            )

    @staticmethod
    def to_uint8_tensor(frames: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(frames):
            raise TypeError(f"Expected floating tensor in [0, 1], got {frames.dtype}.")
        return frames.mul(255.0).round().clamp_(0, 255).to(torch.uint8).contiguous()


class VideoToNumpy(VideoTransform):
    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Get the to numpy transform. Same transform for both train and eval.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable: The to numpy transform.
        """
        if self.backend == "torchvision":
            return self.__class__.to_numpy
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    @staticmethod
    def to_numpy(frames: torch.Tensor) -> np.ndarray:
        """Convert tensor back to numpy array efficiently.

        Args:
            frames: tensor of shape [T, C, H, W] in range [0, 1]
        Returns:
            numpy array of shape [T, H, W, C] in uint8 format
        """
        return (frames.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()

class VideoToPIL(VideoTransform):
    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Get the to PIL transform. Same transform for both train and eval.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable: The to PIL transform.
        """
        if self.backend == "torchvision":
            return self.__class__.to_pil
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    @staticmethod
    def to_pil(frames: torch.Tensor) -> Image.Image:
        """Convert tensor back to PIL Image.

        Args:
            frames: tensor of shape [T, C, H, W] in range [0, 1]
        Returns:
            PIL Image of shape [T, H, W, C] in uint8 format
        """
        # video PIL format?
        return Image.fromarray((frames.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy())
