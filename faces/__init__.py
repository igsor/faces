from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
from PIL import Image as PILImage

from faces.types import (
    BoundingBox,
    FaceEncoding,
    FacePatch,
    FaceProbability,
    Identity,
    Image,
)


class Identifier(ABC):
    """Identify faces."""

    @abstractmethod
    def __call__(self, face_patch: FacePatch) -> Identity:
        """Return the identity of the person in *face_patch*."""


class Detector(ABC):
    """Detect faces."""

    @abstractmethod
    def detect(self, image: Image) -> Iterable[tuple[BoundingBox, FaceProbability]]:
        """Return the bounding boxes and likelihoods of there being a face."""

    @abstractmethod
    def extract(self, image: Image) -> Iterable[tuple[BoundingBox, FacePatch]]:
        """Return the bounding boxes and faces detected in an image."""


class Encoder(ABC):
    """Encode a face patch."""

    @abstractmethod
    def __call__(self, face_patch: FacePatch) -> FaceEncoding:
        """Return the encoding of a *face_path*."""

    @abstractmethod
    def many(self, patches: torch.Tensor) -> torch.Tensor:
        """Return N encodings of face *patches* given as an (N, ...) tensor."""


class Registry(ABC):
    """Face patches and identities storage."""

    @abstractmethod
    def add(self, face_patch: FacePatch, identity: Identity) -> None:
        """Store a face and its identity. Auto-commits."""

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[FacePatch, Identity]]:
        """Iterate over face patches and their identities."""


class Annotate(ABC):
    """Annotate an image with bounding boxes and additional information."""

    @abstractmethod
    def with_probability(
        self,
        image: Image,
        boxes_and_probability: Iterable[tuple[BoundingBox, FaceProbability]],
    ) -> PILImage.Image:
        """Draw bounding boxes and their likelihood of enclosing a face."""

    @abstractmethod
    def with_identity(
        self,
        image: Image,
        boxes_and_identity: Iterable[tuple[BoundingBox, Identity]],
    ) -> PILImage.Image:
        """Draw bounding boxes and their identity."""

    @abstractmethod
    def with_enumeration(
        self, image: Image, boxes: Iterable[BoundingBox], start: int = 0
    ) -> PILImage.Image:
        """Draw bounding boxes and their index in the sequence."""

    @abstractmethod
    def __call__(self, image: Image, boxes: Iterable[BoundingBox]) -> PILImage.Image:
        """Draw bounding boxes."""


class Builder(ABC):
    """Build instances."""

    @property
    @abstractmethod
    def annotate(self) -> Annotate:
        """Return an Annotate instance."""

    @property
    @abstractmethod
    def identifier(self) -> Identifier:
        """Return a Identifier instance."""

    @property
    @abstractmethod
    def encoder(self) -> Encoder:
        """Return an Encoder instance."""

    @property
    @abstractmethod
    def registry(self) -> Registry:
        """Return a Registry instance."""

    @property
    @abstractmethod
    def detector(self) -> Detector:
        """Return a Detector instance."""

    @classmethod
    @abstractmethod
    def from_args(cls, args: argparse.Namespace) -> Builder:
        """Initialize the Builder from argparse arguments."""

    @classmethod
    @abstractmethod
    def from_defaults(cls) -> Builder:
        """Initialize the Builder from default arguments."""
