#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pylab as plt
from PIL import Image as PILImage

from faces import Builder, Identity, Image
from faces.builder import DefaultBuilder


class Main:
    """Detect and identify faces in an image."""

    def main(self, argv) -> None:
        """Perform face detection, identification, or registration action."""
        parser = argparse.ArgumentParser(description=Main.__doc__, prog="faces")
        # generic args
        parser.add_argument(
            "--verbose", action="store_true", default=False, help="increase verbosity"
        )
        parser.add_argument(
            "--device", type=str, default=None, help="cuda device number."
        )
        parser.add_argument(
            "--registry-path",
            type=Path,
            default=Path("~/.faces.pkl").expanduser(),
            help="path to the faces database.",
        )
        # pipeline args
        parser.add_argument(
            "--probability-threshold",
            type=float,
            default=0.9,
            help="only show faces whose likelihood exceeds the given threshold.",
        )
        parser.add_argument(
            "--distance-threshold",
            type=float,
            default=0.9,
            help="only identify faces whose similarity is below the given threshold.",
        )
        # actions
        subparsers = parser.add_subparsers(
            dest="action", required=True, help="choose what to do"
        )
        # detect
        detect_parser = subparsers.add_parser("detect", help="detect faces in images")
        detect_parser.add_argument(
            "--show-probability",
            action="store_true",
            default=False,
            help="show the probability of each face",
        )
        detect_parser.add_argument(
            "images",
            nargs="+",
            type=Path,
            help="images on which to apply face detection.",
        )
        # register
        register_parser = subparsers.add_parser(
            "register", help="add faces to the registry"
        )
        register_parser.add_argument(
            "images",
            nargs="+",
            type=Path,
            help="images on which to apply face detection.",
        )
        # identify
        identify_parser = subparsers.add_parser(
            "identify", help="identify faces in images"
        )
        identify_parser.add_argument(
            "images",
            nargs="+",
            type=Path,
            help="images on which to apply face detection.",
        )

        # parse args
        args = parser.parse_args(argv)

        # setup
        if args.verbose:
            logging.basicConfig(level=logging.INFO)

        builder = DefaultBuilder.from_args(args)

        # take action
        if args.action == "detect":
            detect = (
                self.detect_with_probability if args.show_probability else self.detect
            )
            for path in args.images:
                detect(builder, Image.open(path)).show()
        elif args.action == "identify":
            for path in args.images:
                self.identify(builder, Image.open(path)).show()
        elif args.action == "register":
            for path in args.images:
                self.register(builder, path)
        else:
            raise ValueError(args.action)

    def detect(self, builder: Builder, image: Image) -> PILImage.Image:
        """Return an image where detected faces are highlighted."""
        return builder.annotate(
            image, (box for box, _ in builder.detector.detect(image))
        )

    def detect_with_probability(self, builder: Builder, image: Image) -> PILImage.Image:
        """Return an image where detected faces and their likelihood are highlighted."""
        return builder.annotate.with_probability(image, builder.detector.detect(image))

    def identify(self, builder: Builder, image: Image) -> PILImage.Image:
        """Return an image where detected faces and their identity are highlighted."""
        return builder.annotate.with_identity(
            image,
            (
                (bounding_box, builder.identifier(face_patch))
                for bounding_box, face_patch in builder.detector.extract(image)
            ),
        )

    def register(self, builder: Builder, path: Path) -> None:
        """Extract faces from an image, add them to a face registry.
        If *path* is a file, its filename is used as identity.
        If *path* is a directory, its folder name is used as identity for all
        images it contains.

        In either case, queries the user for the identity if multiple
        faces are detected within an image.

        """

        def _path_to_identity(path: Path) -> Identity:
            return Identity(path.stem.lower().replace("-", "_").replace("_", " "))

        def _add_face(path: Path, label: Path):
            patches = [
                face_patch
                for _, face_patch in builder.detector.extract(Image.open(path))
            ]
            if len(patches) == 1:
                try:
                    builder.registry.add(patches[0], _path_to_identity(label))
                except ValueError as error:
                    print("Skipping face:", error)
            elif len(patches) > 1:
                for face_patch in patches:
                    user_input = ""
                    while not user_input:
                        plt.imshow(
                            ((face_patch.permute(1, 2, 0) * 128 + 128) / 256.0)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        plt.show()
                        print(
                            "Please specify the identity of the previously shown face."
                        )
                        print("Press <Enter> to show the image again")
                        print("Enter -1 to skip this face")
                        print("      -2 to skip this image")
                        print("      -3 to abort and exit")
                        user_input = input(">>> ").strip()

                        if user_input == "-1":
                            user_input = ""
                            break
                        if user_input == "-2":
                            return
                        if user_input == "-3":
                            sys.exit(1)
                    if user_input:
                        try:
                            builder.registry.add(face_patch, Identity(user_input))
                        except ValueError as error:
                            print("Skipping face:", error)

        if path.is_file():
            _add_face(path, path)
        if path.is_dir():
            for child in path.iterdir():
                if child.is_file():
                    _add_face(child, path)


def main(argv=None):
    """Perform face detection, identification, or registration action."""
    Main().main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
