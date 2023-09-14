from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image as PILImage

from faces.utils import preprocess

## simple types
FaceEncoding = torch.Tensor
FacePatch = torch.Tensor
FaceProbability = float
Identity = str


## complex types
@dataclass(frozen=True)
class BoundingBox:
    """A rectangle that supposedly encloses a face."""

    # lower left corner
    lower_left: float
    lower_top: float
    # upper right corner
    upper_left: float
    upper_top: float

    def __post_init__(self):
        assert (
            self.lower_left <= self.upper_left
        ), "lower_left must be smaller than upper_left"
        assert (
            self.lower_top <= self.upper_top
        ), "lower_top must be smaller than upper_top"

    @property
    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return the bounding box as (lower_left, lower_top, upper_left, upper_top)-tuple."""
        return (self.lower_left, self.lower_top, self.upper_left, self.upper_top)


@dataclass(frozen=True)
class Image:
    """An image."""

    image: PILImage.Image

    @classmethod
    def open(
        cls,
        path: Path,
        target_size: int = 1000,
        rotate: Optional[int] = None,
    ) -> Image:
        """Open and preprocess an image at *path*.
        See `faces.utils.preprocess` for the *target_size* and *rotate* parameters.
        """
        return cls(
            preprocess(
                PILImage.open(path),
                target_size=target_size,
                rotate=rotate,
            )
        )
