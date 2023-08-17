import os
from pathlib import Path
from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


with open(Path(os.path.dirname(__file__)) / "requirements.txt") as f:
    required = f.readlines()

setup(
    name="deepseis",
    version="0.0.0",
    author="Megha, Wei, and Nishtha",
    author_email="sai_group@gmail.com",
    description="Deep learnig-based seismological application",
#     license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/srivastavaresearchgroup/DeepSeis",
    packages=find_packages(exclude="tests"),
    python_requires=">=3.7",
    install_requires=required,
)
