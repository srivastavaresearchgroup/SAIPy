import os
from pathlib import Path
from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="saipy",
    author="Megha, Wei, and Nishtha",
    author_email="sai_group@gmail.com",
    description="Deep learnig-based seismological application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/srivastavaresearchgroup/SAIPy",
    license="MIT",
    packages=find_packages(exclude="tests"),
    install_requires=[
	'pytest==6.2.3',
	'numpy==1.21.0',      
	'keyring==22.3.0', 
	'pkginfo==1.7.0',
	'scipy==1.6.2',
	'tensorflow-estimator==2.12.0',	
    'tensorflow~=2.8.0',  
	'keras==2.8.0', 
	'matplotlib==3.7.1', 
	'pandas==1.1.5',
	'tqdm==4.65.0', 
	'h5py==2.10.0', 
	'obspy==1.4.0',
	'jupyter==1.0.0'], 
    
    python_requires='>=3.8',
)
