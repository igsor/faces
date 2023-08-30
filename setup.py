
from setuptools import setup, find_packages
import os

setup(
    # package metadata
    name='faces',
    version='0.23.08',
    author='Matthias Baumgartner',
    author_email='dev@igsor.net',
    description='Face detection, extraction, and identification. An example.',
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    license='BSD',
    license_files=('LICENSE', ),
    url='https://www.igsor.net/faces/',

    # packages
    packages=find_packages(include=['faces']),
    package_dir={'faces': 'faces'},
    # data files are included if mentioned in MANIFEST.in
    include_package_data=True,

    # entrypoints
    entry_points={
        'console_scripts': [
            'faces= faces.main:main',
            ],
        },

    # dependencies
    python_requires=">=3.7",
    install_requires=(
        'facenet_pytorch',
        'numpy',
        'pillow',
        'torch',
        ),
    extras_require={
        'dev': [
            'build',
            'coverage',
            'furo',
            'mypy',
            'pylint',
            'sphinx',
            ],
        },
)
