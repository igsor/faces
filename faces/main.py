#!/usr/bin/env python3

# standard imports
import argparse
import logging
from pathlib import Path
import typing

# external imports
from PIL import Image

# internal imports
from . import detect
from . import utils

# exports
__all__: typing.Sequence[str] = (
    'main',
    )


## code ##

def main(argv=None):
    """
    """
    parser = argparse.ArgumentParser()
    # generic args
    parser.add_argument('--verbose', action='store_true', default=False,
        help='increase verbosity')
    parser.add_argument('--device', type=str, default=None,
        help='cuda device number.')
    subparsers = parser.add_subparsers(dest='action', required=True,
        help='choose what to do')
    # detect
    detect_parser = subparsers.add_parser('detect',
        help='detect faces in images')
    detect_parser.add_argument('--show-probability', action='store_true', default=False,
        help='show face probability.')
    detect_parser.add_argument('--threshold', type=float, default=0.9,
        help='only show faces with higher likelihood.')
    detect_parser.add_argument('images', nargs='+', type=Path,
        help='images to which to apply face detection.')
    # parse args
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if args.action == 'detect':
        # init
        detector = detect.Detect(min_face_prob=args.threshold)
        # detect
        for path in args.images:
            logging.info('Detect faces in %s', path)
            img = Image.open(path)
            utils.annotate(img, detector.detect(img), show_text=args.show_probability).show()
        logging.info('all done')



## main ##

if __name__ == '__main__':
    import sys
    main(sys.argv)

## EOF ##
