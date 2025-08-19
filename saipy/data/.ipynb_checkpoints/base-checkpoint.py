import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from math import sqrt
from random import randint
from tqdm import tqdm
from scipy.signal import stft
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

from obspy import read
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

#__________________________________________________

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

def spec(X):
    X_spec = []
    for i in range(3):
        X_spec.append(np.absolute(stft(X[:,i], 100)[2]))
    
    return np.transpose(np.array(X_spec), (1,2,0))

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
            fig, ax = plt.subplots(1, figsize = [5,4])

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
                y[t0:] = y[t0:] * -0.25 * float(dataset.attrs['source_magnitude'])
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
            if dataset.attrs['trace_category'] == 'noise':
                continue
            p_arr = int(dataset.attrs['p_arrival_sample'])

            Xarr.append(data[p_arr - 32: p_arr + 32, 2])

        return (np.array(Xarr))
    
    def get_creime_rt_data(self,traces = None, training = False):
        
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
                st = randint(0, 5999 - 300)
                end = randint(st +300, 5999)

                Xarr.append(data[st: end, :])
                y = -4 * np.ones((6000))
                yarr.append(y)
            
            else:
                p_arr = int(dataset.attrs['p_arrival_sample'])

                if float(p_arr) < 100:
                    continue
                st = randint(0, p_arr - 100)
                end = randint(p_arr + 100, 5999)

                Xarr.append(data[st: end, :])
                y = -4 * np.ones((6000))
                y[p_arr - st:] = y[p_arr - st:] * -0.25 * float(dataset.attrs['source_magnitude'])
                yarr.append(y)
        if not training:
            return (Xarr, np.array(yarr))
        
        X_train = np.zeros((len(Xarr), 6000, 3))
        
        for i,x in enumerate(Xarr):
            X_train[i,:len(x),:] = x
        X_train_stft = np.array(list(map(spec, list(X_train))))
        return [X_train, X_train_stft], np.array(yarr)
        
    def get_dynapicker_data(self):
        if self.waveforms is None:
            raise ValueError(
                "No waveform data found. Please set metadata_only = False (default option), while loading the dataset.")
        return self.metadata, self.waveforms
    
class INSTANCE:
    def __init__(self, directory=os.getcwd(), metadata_only=False, data='both'):
        self.metadata_n = None
        self.waveforms_n = None
        self.metadata_ev = None
        self.waveforms_ev = None
        if os.path.exists(os.path.join(directory, 'INSTANCE')):
            if data == 'both':
                self.metadata_n = pd.read_csv(os.path.join(directory, 'INSTANCE','metadata_Instance_noise.csv'), low_memory=False)
                self.metadata_ev = pd.read_csv(os.path.join(directory, 'INSTANCE','metadata_Instance_events.csv'), low_memory=False)
            elif data == 'noise':
                self.metadata_n = pd.read_csv(os.path.join(directory, 'INSTANCE','metadata_Instance_noise.csv'), low_memory=False)
            elif data == 'events':
                self.metadata_n = pd.read_csv(os.path.join(directory, 'INSTANCE','metadata_Instance_events.csv'), low_memory=False)
            else:
                raise UnexpectedInputError("Unexpected input to keyword data. The only acceptable inputs are 'noise', 'events' and 'both'")
            if not metadata_only:
                if data == 'both':
                    self.waveforms_ev = h5py.File(os.path.join(directory, 'INSTANCE','Instance_events_counts.hdf5'), 'r')
                    self.waveforms_n = h5py.File(os.path.join(directory, 'INSTANCE','Instance_noise.hdf5'), 'r')
                elif data == 'events':
                    self.waveforms_ev = h5py.File(os.path.join(directory, 'INSTANCE','Instance_events_counts.hdf5'), 'r')
                elif data == 'noise':
                    self.waveforms_ev = h5py.File(os.path.join(directory, 'INSTANCE','Instance_noise.hdf5'), 'r')

        else:
            if directory == 'os.getcwd':
                raise FileNotFoundError(
                    """
                    No folder named 'INSTANCE' found in the current working directory. 
                    If you have the dataset saved in some other directory, please provide the path to the directory using keyword directory.
                    Else, download the dataset following the instructions given in https://github.com/INGV/instance"
                    """)
            else:
                raise FileNotFoundError(
                    """
                    No folder named 'INSTANCE' found in the specified directory.
                    Please, download the dataset following the instructions given in https://github.com/INGV/instance"
                    """)

    def __str__(self):
        if self.metadata_n is not None:
            display(self.metadata_n.head())
        if self.metadata_ev is not None:
            display(self.metadata_ev.head())
        return("INSTANCE Dataset")
        
    def trace_list_noise(self):
        return self.metadata_n['trace_name'].to_list()
    
    def trace_list_events(self):
        return self.metadata_ev['trace_name'].to_list()
    
    def distribution(self, parameter, traces=None, log=False, ax=None, color='slategrey'):
        if traces is None:
            df = self.metadata_ev
        else:
            df = self.metadata_ev[self.metadata_ev['trace_name'].isin(traces)]
            
        if ax is None:
            fig, ax = plt.subplots(1, figsize = [5,4])
        
        ax.hist(df[parameter], log=log, color=color, edgecolor='k')
        ax.set_xlabel(parameter)
        ax.set_ylabel('Frequeny')

    def get_creime_data(self,traces_n=None, traces_ev=None):
        
        if self.waveforms_ev is None and self.waveforms_n is None:
            raise ValueError(
                "No waveform data found. Please set metadata_only = False (default option), while loading the dataset.")
        
        if traces_n is None:
            traces_n = self.trace_list_noise()
        if traces_ev is None:
            traces_ev = self.trace_list_events()
            
        Xarr = []
        yarr = []
        
        pbar = tqdm(total=len(traces_n))
        
        for evi in traces_n:
            pbar.update()
            dataset = self.waveforms_n.get('data/'+str(evi)) 
            data = np.transpose(np.array(dataset))
            st = randint(0, 5999 - 512)

            Xarr.append(data[st: st + 512, :])
            y = -4 * np.ones((512))
            yarr.append(y)
            
        pbar = tqdm(total=len(traces_ev))
        
        for evi in traces_ev:
            pbar.update()
            dataset = self.waveforms_ev.get('data/'+str(evi)) 
            data = np.transpose(np.array(dataset))
            
            p_arr = int(self.metadata_ev[self.metadata_ev['trace_name'] == evi]['trace_P_arrival_sample'])

            if float(p_arr) < 412:
                continue
            t0 = randint(312, 412)
            st = p_arr - t0

            Xarr.append(data[st: st + 512, :])
            y = -4 * np.ones((512))
            y[t0:] = y[t0:] * -0.25 * float(self.metadata_ev[self.metadata_ev['trace_name'] == evi]['source_magnitude'])
            yarr.append(y)

        return shuffle(np.array(Xarr), np.array(yarr))
    
    
    def get_polarcap_data(self,traces = None, training = False):
        
        if self.waveforms_ev is None:
            raise ValueError(
                "No waveform data found. Please set metadata_only = False (default option), while loading the dataset.")
        
        if traces is None:
            traces = self.trace_list_events()
            
        Xarr = []
        yarr = []
        
        pbar = tqdm(total=len(traces))
        
        for evi in traces:
            pbar.update()
            dataset = self.waveforms_ev.get('data/'+str(evi)) 
            data = np.transpose(np.array(dataset))
            
            p_arr = int(self.metadata_ev[self.metadata_ev['trace_name'] == evi]['trace_P_arrival_sample'])

            Xarr.append(data[p_arr - 32: p_arr + 32, 2])
            yarr.append(self.metadata_ev[self.metadata_ev['trace_name'] == evi]['trace_polarity'])
        if not training:
            return (np.array(Xarr), np.array(yarr))
        return (np.array(Xarr), to_categorical((np.array(yarr) == 'positive').astype(int)))
        
    def get_creime_rt_data(self,traces_n = None,traces_ev = None, training = False):
        
        if self.waveforms_ev is None and self.waveforms_n is None:
            raise ValueError(
                "No waveform data found. Please set metadata_only = False (default option), while loading the dataset.")
        
        if traces_n is None:
            traces_n = self.trace_list_noise()
        if traces_ev is None:
            traces_ev = self.trace_list_events()
            
        Xarr = []
        yarr = []
        
        pbar = tqdm(total=len(traces_n))
        
        for evi in traces_n:
            pbar.update()
            dataset = self.waveforms_n.get('data/'+str(evi)) 
            data = np.transpose(np.array(dataset))
            
            st = randint(0, 5999 - 300)
            end = randint(st +300, 5999)

            Xarr.append(data[st: end, :])
            y = -4 * np.ones((6000))
            yarr.append(y)
            
        pbar = tqdm(total=len(traces_ev))
        
        for evi in traces_ev:
            pbar.update()
            dataset = self.waveforms_ev.get('data/'+str(evi)) 
            data = np.transpose(np.array(dataset))   
            p_arr = int(self.metadata_ev[self.metadata_ev['trace_name'] == evi]['trace_P_arrival_sample'])

            if float(p_arr) < 100:
                continue
            st = randint(0, p_arr - 100)
            end = randint(p_arr + 100, 5999)

            Xarr.append(data[st: end, :])
            y = -4 * np.ones((6000))
            y[p_arr - st:] = y[p_arr - st:] * -0.25 * float(self.metadata_ev[self.metadata_ev['trace_name'] == evi]['source_magnitude'])
            yarr.append(y)
        if not training:
            return shuffle(Xarr, np.array(yarr))
        
        X_train = np.zeros((len(Xarr), 6000, 3))
        
        for i,x in enumerate(Xarr):
            X_train[i,:len(x),:] = x
        X_train_stft = np.array(list(map(spec, list(X_train))))
        return shuffle([X_train, X_train_stft], np.array(yarr))
        
    
    def get_dynapicker_data(self, event_type='EQ'):
        if self.waveforms_ev is None and self.waveforms_n is None:
            raise ValueError(
                "No waveform data found. Please set metadata_only = False (default option), while loading the dataset.")
        if event_type == 'EQ':
            return self.metadata_ev, self.waveforms_ev
        else:
            return self.metadata_n, self.waveforms_n
