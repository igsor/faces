import pickle
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Set

from faces import FacePatch, Identity, Registry


class InMemoryRegistry(Registry):
    """Store faces in volatile memory."""

    data: Set[tuple[FacePatch, Identity]]

    def __init__(self):
        self.data = set()

    def add(self, face_patch: FacePatch, identity: Identity) -> None:
        self.data.add((face_patch, identity))

    def __iter__(self) -> Iterator[tuple[FacePatch, Identity]]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class PickleRegistry(Registry):
    """Store faces and identities via pickle."""

    path: Path

    data: Set[tuple[FacePatch, Identity]]

    @classmethod
    def open(cls, path: Path) -> Registry:
        """Open the registry at *path*."""
        if not path.exists():
            return cls(path=path, data=set())
        with open(path, "rb") as registry_file:
            return cls(path=path, data=pickle.load(registry_file)["data"])

    def _save(self) -> None:
        with open(self.path, "wb") as registry_file:
            pickle.dump(
                {
                    "data": self.data,
                },
                registry_file,
            )

    def add(self, face_patch: FacePatch, identity: Identity) -> None:
        self.data.add((face_patch, identity))
        self._save()

    def __iter__(self) -> Iterator[tuple[FacePatch, Identity]]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)
