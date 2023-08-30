
# standard imports
from pathlib import Path
import typing

# external imports
from PIL import Image
from facenet_pytorch import MTCNN
import torch

# internal imports
from . import utils

# exports
__all__: typing.Sequence[str] = (
    'Detect',
    )


class Detect:
    """Detect and extract faces in an image."""

    # target image size after preprocessing.
    _target_size: int

    # minimum box probability to accept it as a face.
    _min_face_prob: float

    # face detection network.
    _mtcnn: MTCNN

    def __init__(
            self,
            # target image size after preprocessing.
            target_size: int = 1000,
            # minimum face probability. Smaller probability will detect more possible faces.
            min_face_prob: float = 0.9,
            # minimum face size. Smaller values will detect more possible faces.
            min_face_size: int = 20,
            # internal probability thresholds. Decrease values to increase the sensitivity.
            thresholds: typing.Tuple[float, float, float] = (0.6, 0.7, 0.7),
            # pyramid scaling factor.
            factor: float = 0.709,
            # torch device. Leave at None to autodetect.
            device: typing.Optional[torch.device] = None,
            ):
        self._target_size = target_size
        self._min_face_prob = min_face_prob
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # initialize the face detection network
        self._mtcnn = MTCNN(
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=factor,
            device=device,
            keep_all=True,
        )

    def detect(
            self,
            img: Image.Image,
            ) -> typing.Iterator[utils.BoundingBox]:
        """Return the bounding box for each detected face."""
        boxes, probs = self._mtcnn.detect(img)
        if boxes is None: # no boxes to return
            return
        for box, prob in zip(boxes, probs):
            if prob >= self._min_face_prob:
                yield utils.BoundingBox(*box, label=prob)

    def extract(
            self,
            img: Image.Image,
            boxes: typing.Sequence[utils.BoundingBox],
            ) -> typing.Iterator[utils.Face]:
        """Return the face for each bounding box."""
        for patch in self._mtcnn.extract(img, np.array([box.as_tuple for box in boxes]), None):
            yield Face(bounding_box=box, patch=patch)

    def from_image(
            self,
            img: Image.Image,
            ) -> typing.Iterator[Face]:
        """Return the faces in *img."""
        # preprocess the image
        #img = utils.preprocess(img, self._target_size)
        # detect faces
        yield from self.extract(img, self.detect(img))

    def from_path(
            self,
            path: Path,
            ) -> typing.Iterator[Face]:
        """Return the faces for the image file at *path*."""
        return self.from_image(Image.open(path))

## EOF ##
