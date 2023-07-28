
from setuptools import setup, find_packages
import os

setup(
    # package metadata
    name='faces',
    version='0.23.07',
    author='Matthias Baumgartner',
    author_email='dev@igsor.net',
    description='An example face detection and person identification pipeline.',
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    license='BSD',
    license_files=('LICENSE', ),
    url='https://www.linux-friends.net/faces/',

    # packages
    packages=find_packages(include=['faces']),
    package_dir={'faces': 'faces'},
    # data files are included if mentioned in MANIFEST.in
    include_package_data=True,

    # entrypoints
    entry_points={
        'console_scripts': [
            'bsfs = faces:main',
            ],
        },

    # dependencies
    python_requires=">=3.7",
    install_requires=(
        'facenet_pytorch',
        'ipykernel',
        'matplotlib',
        'nbformat>=4.2.0',
        'numpy',
        'pandas',
        'pillow',
        'plotly',
        'scikit-learn',
        'torch',
        ),
)
