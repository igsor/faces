
# standard imports
from dataclasses import dataclass
import typing

# external imports
from PIL import Image, ImageFont, ImageDraw
import torch

# constants
FONT = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 30)
EXIF_ORIENTATION_KEY = 274

# exports
__all__: typing.Sequence[str] = (
    'BoundingBox'
    'Face',
    'annotate',
    'preprocess',
    )

## code ##


@dataclass(frozen=True)
class BoundingBox:
    # lower left corner
    x0: float
    y0: float
    # upper right corner
    x1: float
    y1: float
    # label
    label: typing.Optional[typing.Union[str, int, float]] = None

    def __post_init__(self):
        assert self.x0 <= self.x1, "x0 must be smaller than x1"
        assert self.y0 <= self.y1, "y0 must be smaller than y1"

    @property
    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return the bounding box as (x0, y0, x1, y1)-tuple."""
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class Face:
    bounding_box: BoundingBox
    patch: torch.Tensor
    label: typing.Optional[str] = None



@dataclass(frozen=True)
class Color:
    # color channels, in range [0, 255]
    red: int
    green: int
    blue: int
    # alpha channel, in range [0, 255]
    alpha: typing.Optional[int] = 0

    def __post_init__(self):
        assert 0 <= self.red <= 255
        assert 0 <= self.green <= 255
        assert 0 <= self.blue <= 255
        assert 0 <= self.alpha <= 255

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return the color as (red, green, blue, alpha)-tuple."""
        return self.red, self.green, self.blue, self.alpha



def preprocess(
        img: Image.Image,
        target_size: int = 1000,
        rotate: typing.Optional[int] = None,
        ) -> Image.Image:
    """Preprocess an image.

    1. Scale larger side to *target_size*
    2. Rotate by angle *rotate*, or auto-rotate if *rotate=None* (the default).

    """
    # scale image
    if img.size[0] > img.size[1]: # landscape
        img = img.resize((target_size, int(img.height / img.width * target_size)), reducing_gap=3)
    elif img.size[0] < img.size[1]: # portrait
        img = img.resize((int(img.width / img.height * target_size), target_size), reducing_gap=3)

    # rotate image (if need be)
    if rotate is None:
        # auto-rotate according to EXIF information
        img_ori = img.getexif().get(EXIF_ORIENTATION_KEY, None)
        if img_ori == 3:
            img = img.rotate(180, expand=True)
        elif img_ori == 6:
            img = img.rotate(270, expand=True)
        elif img_ori == 8:
            img = img.rotate(90, expand=True)
    elif rotate != 0:
        # manually rotate
        img = img.rotate(rotate, expand=True)

    return img


def annotate(
        img: Image.Image,
        boxes: frozenset[BoundingBox],
        line_width: int = 6,
        show_text: bool = True,
        box_color: Color = Color(255, 0, 0),
        font_color: Color = Color(255, 255, 255, 0),
        float_label_format: str = '{:0.3f}',
        int_label_format: str = '{:d}',
        ) -> Image.Image:
    """Draw bounding boxes and their labels into a copy of *img*. Return the new image.

    Parameters:

    * boxes: Bounding boxes. See `BoundingBox`.
    * line_width: Box line width.
    * show_text: Flag whether to show the box label or not.
    * box_color: Box line color. See `Color`.
    * font_color: Box label color. See `Color`.
    * label_format: Format string for box labels of float or int type.

    """
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(
            box.as_tuple(),
            outline=box_color.as_tuple(),
            width=line_width,
            )
        if show_text and box.label is not None:
            if isinstance(box.label, str):
                label = box.label
            elif isinstance(box.label, int):
                label = int_label_format.format(box.label)
            elif isinstance(box.label, float):
                label = float_label_format.format(box.label)
            else:
                raise TypeError(box.label)
            if len(label) > 0:
                xy = box.x0, box.y0 + line_width
                draw.text(
                    xy,
                    text=label,
                    font=FONT,
                    fill=font_color.as_tuple(),
                    )

    return img

## EOF ##
