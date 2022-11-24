def string_convertor(dd):
    
    dd2 = str(dd).split()
    SNR = []
    for i, d in enumerate(dd2):
        if d != '[' and d != ']':
            
            dL = d.split('[')
            dR = d.split(']')
            
            if len(dL) == 2:
                dig = dL[1]
            elif len(dR) == 2:
                dig = dR[0]
            elif len(dR) == 1 and len(dR) == 1:
                dig = d
            try:
                dig = float(dig)
            except Exception:
                dig = None
                
            SNR.append(dig)
    return(SNR)

import os
import pandas as pd
import h5py
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from tqdm import tqdm
from random import randint

class STEAD:
    def __init__(self, directory = os.getcwd(), metadata_only = False):
        self.metadata = None
        self.waveforms = None
        if os.path.exists(os.path.join(directory, 'STEAD')):
            self.metadata = pd.read_csv(os.path.join(directory, 'STEAD','merged.csv'), low_memory = False)
            self.metadata.snr_db = self.metadata.snr_db.apply(lambda x: np.mean(string_convertor(x)))
            if not metadata_only:
                self.waveforms = h5py.File(os.path.join(directory, 'STEAD','merged.hdf5'), 'r')

        else:
            if directory == 'os.getcwd':
                raise FileNotFoundError(
                    """
                    No folder named 'STEAD' found in the current working directory. 
                    If you have the dataset saved in some other directory, please provide the path to the directory using keyword directory.
                    Else, download the dataset following the instructions given in https://github.com/smousavi05/STEAD"
                    """)
            else:
                raise FileNotFoundError(
                    """
                    No folder named 'STEAD' found in the specified directory.
                    Please, download the dataset following the instructions given in https://github.com/smousavi05/STEAD"
                    """)

    def __str__(self):
        display(self.metadata.head())
        if self.waveforms is None:
            return "Dataset containing {} seismological waveforms.\nMetadata: Available\nWaveform data: Not available".format(len(self.metadata))
        else:
            return "Dataset containing {} seismological waveforms.\nMetadata: Available\nWaveform data: Available".format(len(self.metadata))
        
    def trace_list(self):
        return self.metadata['trace_name'].to_list()
    
    def distribution(self, parameter, traces = None, log = False, ax = None, color = 'slategrey'):
        if traces is None:
            df = self.metadata
        else:
            df = self.metadata[self.metadata['trace_name'].isin(traces)]
            
        if ax is None:
            fig, ax = plt.subplots(1, figsize = [7,5])
        
        if parameter.lower() in ['mag', 'magnitude', 'source_magnitude']:
            df = df[df['trace_category']== 'earthquake_local']
            ax.hist(df['source_magnitude'], bins = np.linspace(-0.5,7.0,16), log = log, color = color, edgecolor = 'k')
            ax.set_xlabel('source_magnitude')
            ax.set_ylabel('Frequeny')
            
        elif parameter.lower() in ['dist', 'epicentral', 'source_distance_km']:
            df = df[df['trace_category']== 'earthquake_local']
            ax.hist(df['source_distance_km'], bins = np.linspace(0,350,8), log = log, color = color, edgecolor = 'k')
            ax.set_xlabel('source_distance_km')
            ax.set_ylabel('Frequeny')
            
        elif parameter.lower() in ['snr', 'snr_db']:
            df = df[df['trace_category']== 'earthquake_local']
            ax.hist(df['snr_db'], log = log, color = color, edgecolor = 'k')
            ax.set_xlabel('snr_db')
            ax.set_ylabel('Frequeny')
            
        else:
            ax.hist(df[parameter], log = log, color = color, edgecolor = 'k')
            ax.set_xlabel(parameter)
            ax.set_ylabel('Frequeny')
            
    def get_creime_data(self,traces = None):
        
        if self.waveforms is None:
            raise ValueError(
                "No waveform data found. Please set metadata_only = False (default option), while loading the dataset.")
        
        if traces is None:
            traces = self.trace_list()
            
        Xarr = []
        yarr = []
        
        pbar = tqdm(total=len(traces))
        
        for evi in traces:
            pbar.update()
            dataset = self.waveforms.get('data/'+str(evi)) 
            data = np.array(dataset)
            
            if dataset.attrs['trace_category'] == 'noise':
                st = randint(0, 5999 - 512)

                Xarr.append(data[st: st + 512, :])
                y = -4 * np.ones((512))
                yarr.append(y)
            
            else:
                p_arr = int(dataset.attrs['p_arrival_sample'])

                if float(p_arr) < 412:
                    continue
                t0 = randint(312, 412)
                st = p_arr - t0

                Xarr.append(data[st: st + 512, :])
                y = -4 * np.ones((512))
                y[t0:] = y[t0:] * -1 * float(dataset.attrs['source_magnitude'])
                yarr.append(y)

        return (np.array(Xarr), np.array(yarr))
    
    
    def get_polarcap_data(self,traces = None):
        
        if self.waveforms is None:
            raise ValueError(
                "No waveform data found. Please set metadata_only = False (default option), while loading the dataset.")
        
        if traces is None:
            traces = self.trace_list()
            
        Xarr = []
        
        pbar = tqdm(total=len(traces))
        
        for evi in traces:
            pbar.update()
            dataset = self.waveforms.get('data/'+str(evi)) 
            data = np.array(dataset)
            
            p_arr = int(dataset.attrs['p_arrival_sample'])

            Xarr.append(data[p_arr - 32: p_arr + 32, 2])

        return (np.array(Xarr))