'''
some codes are cited from https://github.com/omarmohamed15/CapsPhase and 
https://github.com/interseismic/generalized-phase-detection, 
'''
import os
import sys
import numpy as np
import pandas as pd
import torch
import obspy
import obspy.core as oc
from obspy.core import read
from obspy.signal.trigger import trigger_onset
import scipy
from scipy import ndimage,misc
from scipy.signal import butter, lfilter, lfilter_zi
from scipy.special import softmax
from scipy import stats
import scipy.constants
import statistics
import math
import obspy
from obspy import UTCDateTime
import torch
from torch.utils.data import Dataset, DataLoader

def make_stream_stead(dataset):
    '''codes source from https://github.com/smousavi05/STEAD'''
    data = np.array(dataset)
              
    tr_E = obspy.Trace(data=data[:, 0])
    tr_E.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_E.stats.delta = 0.01
    tr_E.stats.channel = dataset.attrs['receiver_type']+'E'
    tr_E.stats.station = dataset.attrs['receiver_code']
    tr_E.stats.network = dataset.attrs['network_code']
    
    tr_N = obspy.Trace(data=data[:, 1])
    tr_N.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_N.stats.delta = 0.01
    tr_N.stats.channel = dataset.attrs['receiver_type']+'N'
    tr_N.stats.station = dataset.attrs['receiver_code']
    tr_N.stats.network = dataset.attrs['network_code']
    
    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.channel = dataset.attrs['receiver_type']+'Z'
    tr_Z.stats.station = dataset.attrs['receiver_code']
    tr_Z.stats.network = dataset.attrs['network_code']

    stream = obspy.Stream([tr_E, tr_N, tr_Z])
    
    return stream

def make_stream_instance(df, h5, line, wftype):
    ''' Codes source from '''
    row = df.iloc[line,:]

    stats = oc.Stats()
    stats.npts = 12000
    stats.sampling_rate = 100.

    sta = row['station_code']
    wav_name = row['trace_name']
    ev_id = row['source_id']
    net = row['station_network_code']

    waveform = h5['data'][row['trace_name']]

    stats.delta = row['trace_dt_s']
    stats.starttime = pd.to_datetime(row['trace_start_time'])
    stats.network = net
    stats.station = sta

    st = oc.Stream()
    for i in range (0,3):
        tr = oc.Trace()
        tr.data = waveform[i]
        tr.stats = stats
        if i == 0:
            tr.stats.channel = row['station_channels'] + 'E'
        if i == 1:
            tr.stats.channel = row['station_channels'] + 'N'
        if i == 2:
            tr.stats.channel = row['station_channels'] + 'Z'
        tr +=tr
        st.append(tr)

    latest_start = np.max([x.stats.starttime for x in st])
    earliest_stop = np.min([x.stats.endtime for x in st])
    st.trim(latest_start, earliest_stop)
    return st, row

def make_stream_array(dataset):
    '''
    input: hdf5 dataset
    output: obspy stream
    
    '''
    data = np.array(dataset)
              
    tr_E = obspy.Trace(data=data[:, 0])
    tr_E.stats.delta = 0.01
    tr_E.stats.channel = 'E'
    
    tr_N = obspy.Trace(data=data[:, 1])
    tr_N.stats.delta = 0.01
    tr_N.stats.channel = 'N'
    
    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.channel = 'Z'

    stream = obspy.Stream([tr_E, tr_N, tr_Z])
    return stream

def band_pass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    zi = lfilter_zi(b, a)
    y,zo = lfilter(b, a, data, zi=zi*data[0])
    return y

def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )
    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )
    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )
    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    if copy:
        return strided.copy()
    else:
        return strided

def softmax_T(x, T=1, copy=True):
    x = x/T
    if copy:
        x = np.copy(x)
    max_prob = np.max(x, axis=1). reshape((-1, 1))
    x -=  max_prob
    x = np.exp(x, x)
    sum_prob = np.sum(x, axis=1).reshape((-1,1))
    x /= sum_prob
    return x

def phase_picking(device, model, stream, bandpass_filter_flag, picker_num_shift, batch_size, fremin, fremax, fo, fs):  
    half_dur = 2.00 # model is pretrained on 4s window waveform
    only_dt = 0.01
    n_win = int(half_dur/only_dt)
    n_feat = 2*n_win
    st = stream.copy()
   
    if bandpass_filter_flag:
        st[0].data = band_pass_filter(st[0].data, fremin, fremax, fs, order=fo)
        st[1].data = band_pass_filter(st[1].data, fremin, fremax, fs, order=fo)
        st[2].data = band_pass_filter(st[2].data, fremin, fremax, fs, order=fo)

        
    # Preparing the sliding windows
    sliding_N = sliding_window(st[0].data, n_feat, stepsize=picker_num_shift)
    sliding_E = sliding_window(st[1].data, n_feat, stepsize=picker_num_shift)
    sliding_Z = sliding_window(st[2].data, n_feat, stepsize=picker_num_shift)
    
    tr_win = np.zeros((sliding_N.shape[0], n_feat, 3))
    tr_win[:,:,0] = sliding_E
    tr_win[:,:,1] = sliding_N
    tr_win[:,:,2] = sliding_Z
    tr_win = tr_win / np.max(np.abs(tr_win), axis=(1,2))[:,None,None]
    
    tr_win = np.reshape(tr_win, (np.shape(tr_win)[0], np.shape(tr_win)[1], np.shape(tr_win)[2])) 
    tr_win_tensor = torch.tensor(tr_win, dtype=torch.float32).permute(0, 2, 1)
    
    dataloader = DataLoader(tr_win_tensor, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=2, drop_last=False)
    win_num = tr_win_tensor.shape[0]
    ts = np.zeros((win_num, 3))
    
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (data) in enumerate(dataloader):
            data = data.to(device)
            if i != len(dataloader)-1:
                ts[i*batch_size:(i+1)*batch_size,:] = model(data).cpu().detach().numpy()
            else:
                ts[i*batch_size:win_num,:] = model(data[:win_num-i*batch_size,:,:]).cpu().detach().numpy()
            torch.cuda.empty_cache()
            
    ts = softmax_T(ts, T = 4)
    prob_P = ts[:, 0]
    prob_S = ts[:, 1]
    
    # Median Filter for Smoothing
    prob_P = ndimage.median_filter(prob_P, size=2) 
    prob_S = ndimage.median_filter(prob_S, size=2)

    pickp = []
    picks = []
   
    p_max = max(list(prob_P))
    if math.isnan(p_max):
        pickp.append(0)
    else:
        pickp.append(list(prob_P).index(p_max))
    s_max = max(list(prob_S[list(prob_P).index(p_max):]))
    if math.isnan(s_max):
        picks.append(0)
    else:
        picks.append(list(prob_P).index(p_max)+ list(prob_S[list(prob_P).index(p_max):]).index(s_max))
    
    pwave = []
    swave = []
    
    pwavetime = []
    swavetime = []
    
    for gp in pickp:
        pwave.append(int(gp*picker_num_shift + 200))

    for gs in picks:
        swave.append(int(gs*picker_num_shift + 200))
        
    return prob_P, prob_S, pwave, swave



