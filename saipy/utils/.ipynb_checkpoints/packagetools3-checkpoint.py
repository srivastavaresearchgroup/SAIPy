import os
import sys
import numpy as np
import pandas as pd
import torch
import obspy
from saipy.data.realdata import *
from obspy import UTCDateTime
from collections import defaultdict
from obspy.signal.filter import bandpass
from saipy.models.creime import CREIME_RT
from saipy.models.dynapicker import *
from saipy.utils.picktools import *
from saipy.utils.visualizations import *
from saipy.models.polarcap import PolarCAP
from obspy.signal.trigger import plot_trigger
from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta, trigger_onset
from obspy.core.trace import Trace
from tqdm import tqdm

def data_windows(X, shift=10):
    X_test = []
    for win in range(1000,len(X),shift):
        length = 1000
        X_test.append(X[win-length:win])
    creime_rt = CREIME_RT()
    _, predictions = creime_rt.predict(X_test)
    return X_test, predictions

def first_non_consecutive(lst):
    for i, j in enumerate(lst, lst[0]):
        if i!=j:
            return j
        
def check_continuity(my_list):
    if not any(a+1!=b for a, b in zip(my_list, my_list[1:])):
        return (True, my_list[0])
    else:
        return (False, first_non_consecutive(my_list))
                  
def polarity_estimation(raw_waveform, dynapicker_output):
    polarcap = PolarCAP()
    windows = []
    polarity_pred = []
    for index in dynapicker_output:
        # predicted P_pick
        X = raw_waveform[int(index[0])-32:int(index[0]) + 32, 2]
        windows.append(X)
        predictions = polarcap.predict(X.reshape(1,X.shape[0], 1))
        polarity_pred.append([predictions[0][0],predictions[0][1]])
    return polarity_pred, windows
            
def detect(raw_waveform, device, idx_list, leng_win, picker_num_shift, batch_size=4, fremin=1, 
                fremax=40, fo=5, fs=100, bandpass_filter_flag=True):
    
    start_sample = idx_list[0] *10
    data = raw_waveform[start_sample:start_sample + leng_win*100,:] 
    stream = make_stream_array(data)
    
    # phase picking
    model = load_model(path = '')
    prob_p, prob_s, pwave, swave = phase_picking(device, model, stream, bandpass_filter_flag, picker_num_shift, 
                                                 batch_size, fremin, fremax, fo, fs)

    p_pick = pwave[0] + start_sample
    s_pick = swave[0] + start_sample
    
    # magnitude estimation
    creime_rt = CREIME_RT()
    _, mag_pred = creime_rt.predict(np.array([raw_waveform[start_sample:min(len(raw_waveform),start_sample + min(6000,leng_win*100)),:]]))
    mag_pred = mag_pred[0][1]
    
    # polarity estimation
    polarcap = PolarCAP()
    X = raw_waveform[int(p_pick)-32:int(p_pick) + 32, 2]
    predictions = polarcap.predict(X.reshape(1, X.shape[0], 1))
    return [p_pick, s_pick, mag_pred, predictions[0][0], predictions[0][1], X]

def plot_save_turkey(raw_waveform, stream, pred, leng_win, save_result, path, file_name, index):
    p_pick, s_pick, mag, polarity, polarity_prob, win = pred
    
    dat = raw_waveform[p_pick - 100: min(len(raw_waveform), p_pick - 100 + leng_win * 100), :]
    fig1 = plot_waveform(dat, times=range(p_pick - 100, min(len(raw_waveform), p_pick - 100 + leng_win * 100)), 
                      P_arr=p_pick, S_arr=s_pick, magnitude=mag)
    fig2 = plot_polarcap_data(win, y_pred = [polarity,polarity_prob])
    fig1.savefig(path+'id_'+str(index)+'_waveform.jpg')
    fig2.savefig(path+'id_'+str(index)+'_polarity.jpg')
#     plt.savefig(path+'id_'+str(i)+'.jpg')
    return True
    
   
def plot_save(raw_waveform, stream, pred, leng_win, save_result, path, file_name):
    p_pick, s_pick, mag, polarity, polarity_prob, win = pred
    
    dat = raw_waveform[p_pick - 100: min(len(raw_waveform), p_pick - 100 + leng_win * 100), :]
    plot_waveform(dat, times=range(p_pick - 100, min(len(raw_waveform), p_pick - 100 + leng_win * 100)), 
                      P_arr=p_pick, S_arr=s_pick, magnitude=mag)
    plot_polarcap_data(win, y_pred = [polarity,polarity_prob])
    
    P_times = stream[0].stats.starttime + p_pick / 100
    S_times = stream[0].stats.starttime + s_pick / 100
    
    if save_result:
        if path is None:
            path = "./"
        if file_name is None:
            raise FileNotFoundError(
                """
                Please provide a file_name to save your file
                """)
        else:
            result_dict = {
                'Station':stream[0].stats.station,
                'Network':stream[0].stats.network,
                'Location':stream[0].stats.location,
                'Channel':stream[0].stats.channel,
                'Trace start time':stream[0].stats.starttime,
                'Trace end time':stream[0].stats.endtime,
                'Pred_P_arrival_sample':[p_pick],
                'Pred_S_arrival_sample':[s_pick],
                'Pred_P_arrival_time':[P_times],
                'Pred_S_arrival_time':[S_times],
                'Pred_magnitude':[mag],
                'Pred_polarity':[polarity],
                'Pred_polarity_prob':[polarity_prob]
            }
            df_result = pd.DataFrame(result_dict) 
            if not os.path.exists(path):
                os.makedirs(path)
                
            result_file = os.path.join(path, file_name)
            if os.path.exists(result_file):
                exist_file = pd.read_csv(result_file)
                if int(p_pick) not in exist_file['Pred_P_arrival_sample'].values.tolist():
                    df_result.to_csv(result_file, mode='a', index=False, header=False)
            else: 
                df_result.to_csv(result_file, index=False)\
                
def monitor(wsp, network, station, location, channel, start_time, end_time, device, leng_win, shift=10, 
               picker_num_shift=1, save_result=False, path=None, file_name=None):
    
    print("Downloading data...")
    st = waveform_download(wsp, net=network, sta=station, loc=location, chan=channel, starttime=start_time, endtime=end_time)
    stream =  st
    print(st)
    print("Pre-processing data...")
    resample_stream, raw_waveform = preprocessing(st)
    print(resample_stream)
    plot_waveform(raw_waveform)
    print("Monitoring...")
    
    window_waveform, creime_output = data_windows(raw_waveform, shift=shift)
    
    win_id1 = [] # all windows labeled as earthquqke
    for i in range(len(creime_output)):
        if creime_output[i][0] == 1:
            win_id1 += [i]
    
    # window selection
    win_id2 = [] 
    i, len_win = 0, len(win_id1)
    
    P_picks, S_picks, mags, polarity_pred = [], [], [], []
    while i < len_win:
        if i+5 <= len_win-1: 
            index_list = win_id1[i:i+5]
            flag, id = check_continuity(index_list)
            if flag == True:
                win_id2 += [index_list]
                # predcition
                pred = detect(raw_waveform, device, index_list, leng_win, picker_num_shift, 
                                   batch_size=4, fremin=1, fremax=40, fo=5, fs=100, bandpass_filter_flag=True)
                
                print('Result: P_arrival_sample={}, S_arrival_sample={}, Magnitude={:.2f}, Polarity={}, Polarity_proba={:.3f}'.\
                      format(pred[0], pred[1], pred[2] if pred[2] is not None else 0, pred[3], pred[4]))
                
                P_picks.append(pred[0])
                S_picks.append(pred[1])
                mags.append(pred[2] if pred[2] is not None else 0)
                polarity_pred.append([pred[3], pred[4]])
                
                # visualization
                plot_save(raw_waveform, stream, pred, leng_win, save_result, path, file_name)
                
                next_ind = id + win_id1.index(id) + int(leng_win*10)
                if len(np.array(win_id1)[np.array(win_id1) > next_ind]) == 0:
                    break
                i = win_id1.index(np.array(win_id1)[np.array(win_id1) > next_ind][0])
            else:
                i = win_id1.index(id)+1
                
    outputs = {
        'P_picks': P_picks, 
        'S_picks': S_picks, 
        'magnitudes': mags, 
        'polarities': polarity_pred
    }
        
    return outputs

def phase_pick_func(raw_waveform, creime_output, device, leng_win=30, picker_num_shift=1, batch_size=4,
                    fremin=1, fremax=40, fo=5, fs=100, bandpass_filter_flag=True):
    win_num = len(creime_output)
    win_id1 = [] # all windows labeled as earthquqke
    for i in range(len(creime_output)):
        if creime_output[i][0] == 1:
            win_id1 += [i]
    win_id2 = [] 
    len_win = len(win_id1)
    i =  0
    while i<len_win:
        if i+5 <= len_win-1: 
            index_list = win_id1[i:i+5]
            flag, id = check_continuity(index_list)
            if flag == True:
                win_id2 += [index_list]
                next_ind = id + win_id1.index(id) + int(leng_win*10)
                if len(np.array(win_id1)[np.array(win_id1) > next_ind]) == 0:
                    break
                i = win_id1.index(np.array(win_id1)[np.array(win_id1) > next_ind][0])

            else:
                i = win_id1.index(id)+1
                
    # phase picking
    dynapicker_outputs = []
    model = load_model(path = '')
    for idx_list in win_id2:
        p_pick = 0
        s_pick = 0
        start_sample = idx_list[0] *10
        data = raw_waveform[start_sample:start_sample + leng_win*100,:] 
        stream =  make_stream_array(data)
        prob_p, prob_s, pwave, swave = phase_picking(device, model, stream, bandpass_filter_flag, picker_num_shift, 
                                                     batch_size, fremin, fremax, fo, fs)
       
        p_pick = pwave[0] + start_sample
        s_pick = swave[0] + start_sample
        creime_rt = CREIME_RT()
        _, mag_pred = creime_rt.predict(np.array([raw_waveform[start_sample:min(len(raw_waveform),
                                        start_sample + min(6000,leng_win*100)),:]]))
        mag_pred = mag_pred[0][1]
        dynapicker_outputs.append([p_pick, s_pick, mag_pred])
    return dynapicker_outputs
    
def monitoring2(station, stream, device, leng_win, shift = 10, picker_num_shift = 10, 
                save_result = False, path = None, file_name = None):
    
    st = stream
    resample_stream, raw_waveform = preprocessing(st)

    window_waveform, creime_output = data_windows(raw_waveform, shift = shift)
    
    dynapicker_output = phase_pick_func(raw_waveform, creime_output, device, picker_num_shift=picker_num_shift,
                                       fremin = 5, fremax = 40, bandpass_filter_flag = True)
    polarity_pred, wins = polarity_estimation(raw_waveform, dynapicker_output)

    os.makedirs(path)
    for i in range(len(dynapicker_output)):
        p_pick, s_pick, mag = dynapicker_output[i]
        X = raw_waveform[int(p_pick)-32:int(p_pick) + 32, 2]
        pred = [p_pick, s_pick, mag, polarity_pred[i][0], polarity_pred[i][1], X]
         # visualization
        plot_save_turkey(raw_waveform, stream, pred, leng_win, save_result, path, file_name, index=i)
    
#     P_picks = [dynapicker_output[i][0] for i in range(len(dynapicker_output))]
#     S_picks = [dynapicker_output[i][1] for i in range(len(dynapicker_output))]
#     mags = [dynapicker_output[i][2] for i in range(len(dynapicker_output))]
    
#     for p, s, m, pol, w in zip(P_picks, S_picks, mags, polarity_pred, wins):
#         dat = raw_waveform[p - 100: min(len(raw_waveform), p - 100 + leng_win * 100), :]

#     outputs = {
#         'P_picks': P_picks, 
#         'S_picks': S_picks, 
#         'magnitudes': mags, 
#         'polarities': polarity_pred
#     }
        
#     P_times = [stream[0].stats.starttime + p / 100 for p in P_picks]  
#     S_times = [stream[0].stats.starttime + s / 100 for s in S_picks] 
    
#     if save_result:
#         if path is None:
#             path = "./"
#         if file_name is None:
#             raise FileNotFoundError(
#                 """
#                 Please provide a file_name to save your file
#                 """)
#         else:
#             result_dict = {
#                 'Station':stream[0].stats.station,
#                 'Network':stream[0].stats.network,
#                 'Location':stream[0].stats.location,
#                 'Channel':stream[0].stats.channel,
#                 'Trace start time':stream[0].stats.starttime,
#                 'Trace end time':stream[0].stats.endtime,
#                 'Pred_P_arrival_sample':P_picks,
#                 'Pred_S_arrival_sample':S_picks,
#                 'Pred_P_arrival_time':P_times,
#                 'Pred_S_arrival_time':S_times,
#                 'Pred_magnitude':mags,
#                 'Pred_polarity':[i[0] for i in polarity_pred],
#                 'Pred_polarity_prob':[i[1] for i in polarity_pred]
#             }
#             df_result = pd.DataFrame(result_dict) 
#             if not os.path.exists(path):
#                 os.makedirs(path)
#             df_result.to_csv(os.path.join(path,file_name), index=False)
                
#     return outputs
               
# STA/LTA method
def classic_picking(trigger_type, trace, nsta, nlta, thr_on, thr_off, plotFlag):
    df = trace.stats.sampling_rate
    if trigger_type == 'classic_sta_lta':
        trigger_func = classic_sta_lta
    elif trigger_type == 'recursive_sta_lta':
        trigger_func = recursive_sta_lta
    cft = trigger_func(trace.data, int(nsta * df), int(nlta * df))       
    if plotFlag:
        plot_trigger(trace, cft, thr_on, thr_off)
    return cft
