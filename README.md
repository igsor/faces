
# Faces - An example face detection and identification module

## Introduction

This is a python module to detect faces in an image, and to identify each of them.

Face identification works in two steps.
First, we detect faces in the image.
At this stage, we don't care about who's face we find, the goal is to find all faces in the image.
Second, we identify a previously found face, i.e. we want to show the name of the depicted person.
To be able to do this,
we need a database that contains people's names and some example images of their face.
The idea is that we can compare the query face to all example images from that database
and use the name associated with the closest match.

The module is intended as an educational demonstration of face identification.
The heavy lifting is done by FaceNet [^1]. 
Specifically, this module is built on the [pytorch](https://pytorch.org) implementation of FaceNet [^2].
FaceNet comes with two components.
The first component can detect faces in an image (i.e., the first step of our pipeline).
This component (MTCNN) was trained on millions of images to detect and locate faces.

But simply comparing images of faces to each other (i.e., the second step of our pipeline)
would not give us good results.
The reason is that images are taken at different angles or light conditions,
and the person's appearance might differ across images due to hair styles or worn accessories.
Instead, we need to convert the images into a format that is independent of such factors.
That format should only express the core characteristics of a face, like the eye color,
face geometry (nose width, jawline, eye distance), etc.
We call this an encoding.
The second component FaceNet does just that.
It uses another deep learning model (Inception-ResNet) which was again trained on a huge
number of faces to carve out the "essence" of a face.

This module uses FaceNet to detect, extract, and compare images,
and offers the convenience and tooling around the bare FaceNet components.
For example, you can detect and identify faces with a few simple commands,
or you can tune the sensitivity of these components through the accompanying notebooks
to get better performance on your specific image library.


## Installation

Set up a virtual environment:

    $ virtualenv env
    $ source env/bin/activate

**NOTE**: install torch according to https://pytorch.org before continuing!

Install faces as editable from the git repository:

    $ git clone https://github.com/igsor/faces
    $ cd faces
    $ pip install -e .

If you want to develop (*dev*) faces with the respective extras:

    $ pip install -e ".[dev]"

To ensure code style discipline, run the following commands:

    $ isort faces
    $ black faces
    $ coverage run ; coverage html ; xdg-open .htmlcov/index.html
    $ pylint faces
    $ mypy

To build the package, do:

    $ python -m build

To run only the tests (without coverage), run the following command from the **test folder**:

    $ python -m unittest

To build the documentation, run the following commands from the **docs folder**:

    $ sphinx-apidoc -f -o source/api ../faces/ --module-first -d 1 --separate
    $ make html
    $ xdg-open build/html/index.html



## References

[^1]: F. Schroff, D. Kalenichenko, J. Philbin. FaceNet: A Unified Embedding for Face Recognition and Clustering, arXiv:1503.03832, 2015. [PDF](https://arxiv.org/pdf/1503.03832.pdf)
[^2]: [https://github.com/timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)




