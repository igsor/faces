import unittest
from pathlib import Path
from tempfile import mkstemp
from typing import Iterable, Tuple

import numpy as np
import torch

from faces import FacePatch, Identity
from faces.registry import InMemoryRegistry, PickleRegistry


class TestInMemoryRegistry(unittest.TestCase):
    def _initialize_registry(
        self,
    ) -> Tuple[InMemoryRegistry, Iterable[Identity], Iterable[FacePatch]]:
        registry = InMemoryRegistry()

        queries = (
            "eric-idle.npy",
            "graham-chapman.npy",
            "john-cleese.npy",
            "michael-palin.npy",
            "terry-gilliam.npy",
            "terry-jones.npy",
        )
        patches = [
            FacePatch(np.load(Path(__file__).parent / "data" / "patches" / query))
            for query in queries
        ]

        for identity, patch in zip(queries, patches):
            registry.add(patch, identity)

        return registry, queries, patches

    def test_add(self) -> None:
        registry, queries, patches = self._initialize_registry()
        self.assertEqual(len(registry.data), 6)
        self.assertSetEqual(set(registry.data), set(zip(patches, queries)))

    def test_query(self) -> None:
        registry, queries, patches = self._initialize_registry()
        self.assertSetEqual(set(registry), set(zip(patches, queries)))

    def test_len(self) -> None:
        registry = InMemoryRegistry()
        # new registry
        self.assertEqual(len(registry), 0)
        # loaded registry
        registry, _, _ = self._initialize_registry()
        self.assertEqual(len(registry), 6)


class TestPickleRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.registry_base_path = Path(mkstemp(prefix="faces-test-")[1])
        self.registry_path = Path(str(self.registry_base_path) + "-missing")

    def tearDown(self) -> None:
        self.registry_base_path.unlink(missing_ok=True)
        self.registry_path.unlink(missing_ok=True)

    def test_open(self) -> None:
        # open new registry
        registry = PickleRegistry.open(self.registry_path)
        self.assertIsInstance(registry, PickleRegistry)
        self.assertEqual(len(registry.data), 0)
        # open non-empty registry
        registry = PickleRegistry.open(
            Path(__file__).parent / "data" / "registry" / "faces.pkl"
        )
        self.assertEqual(len(registry.data), 4)

    def _initialize_registry(
        self,
    ) -> Tuple[PickleRegistry, Iterable[Identity], Iterable[FacePatch]]:
        registry = PickleRegistry.open(self.registry_path)

        queries = (
            "eric-idle.npy",
            "graham-chapman.npy",
            "john-cleese.npy",
            "michael-palin.npy",
            "terry-gilliam.npy",
            "terry-jones.npy",
        )
        patches = [
            FacePatch(np.load(Path(__file__).parent / "data" / "patches" / query))
            for query in queries
        ]

        for identity, patch in zip(queries, patches):
            registry.add(patch, identity)

        return registry, queries, patches

    def test_add(self) -> None:
        registry, queries, patches = self._initialize_registry()
        # registry has been modified
        self.assertEqual(len(registry.data), 6)
        self.assertSetEqual(set(registry.data), set(zip(patches, queries)))
        # registry has been saved
        reloaded = PickleRegistry.open(self.registry_path)
        self.assertEqual(len(registry.data), 6)
        self.assertSetEqual(set(registry.data), set(zip(patches, queries)))

    def test_query(self) -> None:
        # new registry
        registry, queries, patches = self._initialize_registry()
        self.assertSetEqual(set(registry), set(zip(patches, queries)))
        # loaded registry
        registry = PickleRegistry.open(
            Path(__file__).parent / "data" / "registry" / "faces.pkl"
        )
        self.assertEqual(len(registry), 4)

    def test_len(self) -> None:
        # new registry
        self.assertEqual(len(PickleRegistry.open(self.registry_path)), 0)
        # loaded registry
        registry = PickleRegistry.open(
            Path(__file__).parent / "data" / "registry" / "faces.pkl"
        )
        self.assertEqual(len(registry), 4)


if __name__ == "__main__":
    unittest.main()
