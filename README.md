# SAIPy
Seismology has witnessed significant advancements in recent years with the application of deep learning methods to address a broad range of problems. These techniques have demonstrated their remarkable ability to effectively extract statistical properties from extensive datasets, surpassing the capabilities of traditional approaches to an extent. In this repository we present SAIPy, an open-source Python package developed for fast seismic waveform data processing by implementing deep learning. SAIPy offers solutions for multiple seismological tasks such as earthquake detection, magnitude estimation, seismic phase picking, and polarity identification. This brings together the capabilities of previously published models such as [CREIME](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022JB024595), [DynaPicker](https://arxiv.org/abs/2211.09539v1?trk=public_post_main-feed-card_feed-article-content) and [PolarCAP](https://www.sciencedirect.com/science/article/pii/S2666544122000247) and introduces upgraded versions of previously published models such as CREIME_RT capable of identifying earthquakes with an accuracy above 99.8% and a root mean squared error of 0.38 unit in magnitude estimation. These upgraded models outperform state-of-the-art approaches like the Vision Transformer network. SAIPy provides an API that simplifies the integration of these advanced models with benchmark datasets like STEAD and INSTANCE.

# Version 2.0.0 Release Notes
In this release, the package uses CREIME_RT, DynaPicker_v2, and PolarCAP for the analysis of seismograms in a single-station setting. Optionally, this version introduces a new approach for analyzing the results of these models across multiple stations, making use of a seismic network database.
For further details and usage instructions, please refer to the documentation (see below).

## Installation
To install this package clone this repository using 

    git clone https://github.com/srivastavaresearchgroup/SAIPy.git

  It is recommended that you create a virtual environment to install SAIPy. To do this, create a folder, create a virtual environment in that folder, and activate the environment:
     
    mkdir SAIPy_venv
    python3 -m venv SAIPy_venv
    source SAIPy_venv/bin/activate
  
  Then change working directory to SAIPy and run the following command:

    python3 -m pip install .

  Make sure to use the correct version of python installed in your system for the above command (example for python3).

## Documentation
The documentation for using the last updates of the SAIPy package is in [SAIPy_Documentation_v2.0.0.pdf](https://github.com/srivastavaresearchgroup/SAIPy/blob/main/SAIPy_Documentation_v2.pdf). Additionally, you can test the examples 7, 8 and 9, that we provide in (https://github.com/srivastavaresearchgroup/SAIPy/blob/main/examples_v2.0.0/).


## Reach out to us
We strive to constantly improve and update SAIPy.

Should you have any queries or suggestions do not hesitate to contact the authors:
* Megha Chakraborty (chakraborty@fias.uni-frankfurt.de)
* Wei Li (wli@fias.uni-frankfurt.de)
* Claudia Quinteros Cartaya (quinteros@fias.uni-frankfurt.de / quinterosclaudia@gmail.com)
* Jonas Köhler (jkoehler@fias.uni-frankfurt.de)
* Johannes Faber (faber@fias.uni-frankfurt.de)
* Nishtha Srivastava-Team leader (srivastava@fias.uni-frankfurt.de)
  
## Citation
Cite as:

Wei Li, Megha Chakraborty, Claudia Quinteros-Cartaya, Jonas Köhler, Johannes Faber, Men-Andrin Meier, Georg Rümpker, Nishtha Srivastava. SAIPy: A Python package for single-station earthquake monitoring using deep learning. Computers & Geosciences, Volume 192, 2024, 105686, ISSN 0098-3004, https://doi.org/10.1016/j.cageo.2024.105686.
