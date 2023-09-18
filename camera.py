#!/usr/bin/env python3

import logging
from collections import namedtuple
from datetime import datetime
from tempfile import mkstemp

import cv2
import numpy as np

from faces import Identity, Image
from faces.builder import DefaultBuilder
from faces.main import Main

logging.basicConfig(level=logging.INFO)

WINDOW_NAME = "continuous face identification"

VideoFrame = namedtuple("VideoFrame", ["rval", "frame"])

# initialize faces module
builder = DefaultBuilder.from_defaults()
# initialize output window
cv2.namedWindow(WINDOW_NAME)
# initialize video capture
vc = cv2.VideoCapture(0)

identified_in_session = set()

while True:
    # grab frame
    if not (video_frame := VideoFrame(*vc.read())).rval:
        break

    # load the image
    image = Image.from_array(video_frame.frame)

    # identify faces in the image
    extracts = [
        (bounding_box, face_patch, builder.identifier(face_patch))
        for bounding_box, face_patch in builder.detector.extract(image)
    ]

    # faces w/o identity
    unidentified = [
        patch
        for _, patch, identity in extracts
        if identity == builder.identifier.restklasse
    ]

    # log freshly identified people
    identified = {
        identity
        for _, _, identity in extracts
        if identity != builder.identifier.restklasse
    }
    for name in identified - identified_in_session:
        logging.info(f"found {name}")
    identified_in_session |= identified

    # annotate the image show it
    cv2.imshow(
        WINDOW_NAME,
        np.array(
            builder.annotate.with_identity(
                image, ((bbox, identity) for bbox, _, identity in extracts)
            )
        ),
    )

    if (key := cv2.waitKey(20)) == 27:  # exit on ESC
        break
    elif key == 32:  # save image on spacebar
        timestamp = datetime.now().isoformat()
        image.image.save(
            mkstemp(prefix=f"faces-capture-{timestamp}-", suffix=".jpg")[1]
        )
    elif key == 13:  # add face to database on ENTER
        if not unidentified:
            logging.error("requires one unidentified face to have been detected")
        elif len(unidentified) != 1:
            logging.error("can only add one face at a time")
        else:
            # one unidentified face - ask user for identity
            user_input = ""
            while not user_input:
                print("Please specify the identity of the face.")
                print("Enter -1 to skip this face")
                user_input = input(">>> ").strip()
                if user_input == "-1":
                    user_input = ""
                    break

            if user_input:
                try:
                    (face_patch,) = unidentified
                    builder.registry.add(face_patch, Identity(user_input))
                    builder.reload()
                except ValueError as error:
                    logging.error("Skipping face: %s", error)

# cleanup
vc.release()
cv2.destroyWindow(WINDOW_NAME)
