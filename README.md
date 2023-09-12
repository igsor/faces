
# Faces


## Installation

Set up a virtual environment:

    $ virtualenv env
    $ source env/bin/activate

# NOTE: install torch according to https://pytorch.org before continuing!

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

