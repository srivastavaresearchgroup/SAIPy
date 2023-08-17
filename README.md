# SAIPy
Seismology has witnessed significant advancements in recent years with the application of deep learning methods to address a broad range of problems. These techniques have demonstrated their remarkable ability to effectively extract statistical properties from extensive datasets, surpassing the capabilities of traditional approaches to an extent. In this repository we present SAIPy, an open-source Python package developed for fast seismic waveform data processing by implementing deep learning. SAIPy offers solutions for multiple seismological tasks such as earthquake detection, magnitude estimation, seismic phase picking, and polarity identification. This brings together the capabilities of previously published models such as [CREIME](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022JB024595), [Dynapicker](https://arxiv.org/abs/2211.09539v1?trk=public_post_main-feed-card_feed-article-content) and [PolarCAP](https://www.sciencedirect.com/science/article/pii/S2666544122000247) and introduces upgraded versions of previously published models such as CREIME_RT capable of identifying earthquakes with an accuracy above 99.8% and a root mean squared error of 0.38 unit in magnitude estimation. These upgraded models outperform state-of-the-art approaches like the Vision Transformer network. SAIPy provides an API that simplifies the integration of these advanced models with benchmark datasets like STEAD and INSTANCE. The package can be implemented on continuous waveforms and has the potential to be used for real-time earthquake monitoring to enable timely actions to mitigate the impact of seismic events.

## Documentation:

### deepseis.data.base.STEAD(directory = os.getcwd(), metadata_only = False)
A STEAD dataset object.

Parameters: directory (str, optional): path to directory where ‘STEAD’ folder can be found; defaults to current working directory\
            metadata_only (bool, optional): if True only metadata is read to the object
            
Attributes: metadata: pandas.DataFrame containing metadata\
            waveforms: h5py object containing waveforms
