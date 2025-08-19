import numpy as np
import os
import pandas as pd
import sys
import time
import torch

from collections import defaultdict
from tqdm import tqdm

import obspy
from obspy import read
from obspy import UTCDateTime
from obspy.core.trace import Trace
from obspy.signal.filter import bandpass
from obspy.signal.trigger import plot_trigger
from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta, trigger_onset

from saipy.data.realdata import *
from saipy.models.creime import CREIME_RT
from saipy.models.dynapicker import *
from saipy.models.polarcap import PolarCAP
from saipy.utils.picktools import *
from saipy.utils.visualizations import *
from saipy.user_settings import date_format_object

#_____________________________

def windows_for_creime(modpath, X, shift=10):
    X_test = []
    len_init = 1000 # lenght of windows in samples (default:1000 samples, 10 seconds). 
                    # Maximum allowed 6000 samples (60 seconds).
    for win in range(len_init,len(X),shift):  
        X_test.append(X[win-len_init:win])
    creime_rt = CREIME_RT(modpath)
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
                  
def polarity_classification(prepro_wave_array, dynapicker_output, path):
    polarcap = PolarCAP(path)
    windows = []
    polarity_pred = []
    for index in dynapicker_output:
        # predicted P_pick
        X = prepro_wave_array[int(index[0])-32:int(index[0]) + 32, 2]
        windows.append(X)
        predictions = polarcap.predict(X.reshape(1,X.shape[0], 1))
        polarity_pred.append([predictions[0][0],predictions[0][1]])
    return polarity_pred, windows
      
def save_results(path, file_name, network, station, location, start_time, end_time, pred):
    p_pick, s_pick, mag, polarity, polarity_prob = pred
    
    P_times = start_time + p_pick / 100
    S_times = start_time + s_pick / 100
    
    if path is None:
            path = "./"
    if file_name is None:
        raise FileNotFoundError(
            """
            Please provide a file_name to save your file
            """)
    else:
        result_dict = {
            'Station':[station],
            'Network':[network],
            'Location':[location],
            'Trace start time':[start_time],
            'Trace end time':[end_time],
            'Pred_P_arrival_sample':[p_pick],
            'Pred_S_arrival_sample':[s_pick],
            'Pred_P_arrival_time':[P_times],
            'Pred_S_arrival_time':[S_times],
            'Magnitude':[mag],
            'Pred_polarity':[polarity],
            'Pred_polarity_prob':[polarity_prob]
        }
        #'Channel':[channel],
        df_result = pd.DataFrame(result_dict) 
        if not os.path.exists(path):
            os.makedirs(path)

        result_file = os.path.join(path, file_name)
        if os.path.exists(result_file):
            exist_file = pd.read_csv(result_file)
            if int(p_pick) not in exist_file['Pred_P_arrival_sample'].values.tolist():
                df_result.to_csv(result_file, mode='a', index=False, header=False)
        else: 
            df_result.to_csv(result_file, index=False)

def monitor(prepro_stream, prepro_wave_array, device, station_id,
            date, time_id, len_win, shift=10, picker_num_shift=10, detection_windows=5,
            sampling=100, model_path='', save_result=True, outpath='./'):
    """
    Monitors earthquake events by Saipy: Creime_rt, dynapicker_v2, and polarCAP.

    Parameters:
        prepro_stream     (Stream) : ObsPy stream object containing preprocessed seismic data.
        prepro_wave_array (ndarray): Processed seismic waveform array.
        device           (str)     : Computational device ('cpu' or 'cuda' for GPU).
        station_id       (str)     : ID name of the station.
        date             (str)     : Date as the filename format.
        time_id          (str)     : Start time as the filename format.
        len_win          (int)     : Length of the scanning time window in seconds.
        shift            (int)     : Step size for detection in samples (default: 10).
        picker_num_shift (int)     : Step size for phase picking in samples (default: 10).
        detection_windows (int)    : Number of consecutive positive detections to confirm an event.
        sampling         (int)     : Sampling rate in Hz (default: 100).
        model_path       (str)     : Path to the trained models.
        save_result      (bool)    : Whether to save results (default: True).
        outpath          (str)     : Directory to save results.

    Returns:
        dict: A dictionary containing:
            - 'Start_time' : UTC start time of the stream.
            - 'P_picks'    : List of P-wave arrival times.
            - 'S_picks'    : List of S-wave arrival times.
            - 'init_window': List of start window detection, sample. 
            - 'picks_prob' : List of P and S pick probabilities.
            - 'magnitudes' : List of estimated magnitudes.
            - 'polarities' : List of polarity predictions with probabilities.
    """

    # Initialize variables
    len_win_id = int(len_win * sampling / shift)
    start_time = prepro_stream[0].stats.starttime

    print(f"\n** Monitoring data... **")
    print(f"Earthquake scan time window: {len_win} seconds")
    print(f"Stepsize for detection: {shift} samples")
    print(f"Earthquake detected with {detection_windows} consecutive positive detections")
    print(f"Stepsize for picking: {picker_num_shift} samples\n")
    
    if save_result:
        # Name of specific folders and output file (csv format)
        dirout_specific = os.path.join(outpath, station_id, f"{date}_{time_id}")
        out_file = f"results_{station_id}_{date}_{time_id}"

        # Remove existing output file if exists
        path_file = os.path.join(dirout_specific, f"{out_file}.csv")
        if os.path.exists(path_file):
            print(f"Removing the old output file {out_file}.csv\n")
            os.remove(path_file)

    # **1) Running CREIME for event detection**
    print("1) Running Creime_rt for event detection and magnitude")
    itime1 = time.time()
    window_waveform, creime_output = windows_for_creime(model_path, prepro_wave_array, shift=shift)
    print(f'Time elapsed: {time.time() - itime1:.2f} seconds')

    # Extract windows labeled (detections)
    win_id1 = [i for i, output in enumerate(creime_output) if output[0] == 1]
    len_id1 = len(win_id1)

    print(f'Number of detected earthquake windows: {len_id1}')
    P_picks, S_picks, init_win, picks_prob, mags, polarity_pred = [], [], [], [], [], []

    if len_id1 == 0:
        print("No events detected")
    else:
        icount, idet = 0, 0
        while len_id1 - icount >= detection_windows:
            index_list = win_id1[icount:icount + detection_windows]
            flag, id1_0f = check_continuity(index_list)
            
            print("\n___________________________________________")
            print(f"\nDetected event window indices: {index_list}")
            print(f"Continuity check result: {flag}")
            
            if flag:
                idet += 1
                print(f"\n____________ Event: {idet} _____________")
                # **Magnitude Estimation**
                magnitudes = creime_output[id1_0f:index_list[-1] + 1]
                mag_pred = round(np.mean([m[1] for m in magnitudes]), 1)

                if (id1_0f * shift) + (detection_windows * sampling) - 1 <= len(prepro_wave_array):                    
                    # **2) Running DynaPicker for Phase Picking**
                    print("\n2) Running DynaPicker for phase picking")
                    itime2 = time.time()
                    start_sample = id1_0f * shift
                    init_ref_pick = start_sample - 50 # from 0.5 seconds before
                    final_pick = start_sample + ((len_win) * sampling)
                    X_data = prepro_wave_array[init_ref_pick:final_pick, :]
                    model = load_dynapicker(path=model_path)
                    prob_p, prob_s, pwave, swave = phase_picking2(device, model, X_data,
                                                                  picker_num_shift=picker_num_shift, batch_size=4)

                    p_pick, s_pick = pwave[0] + init_ref_pick, swave[0] + init_ref_pick
                    p_utc, s_utc = start_time + (p_pick / sampling), start_time + (s_pick / sampling)
                    print(f'Time elapsed: {time.time() - itime2:.2f} seconds')

                    # **3) Running PolarCAP for Polarity Estimation**
                    print("\n3) Running PolarCAP for polarity estimation")
                    itime3 = time.time()
                    polarcap = PolarCAP(model_path)
                    X = prepro_wave_array[int(p_pick) - 32:int(p_pick) + 32, 2]
                    predictions = polarcap.predict(X.reshape(1, X.shape[0], 1))
                    pol, prob_pol = predictions[0][0], predictions[0][1]
                    print(f"Time elapsed: {time.time() - itime3:.2f} seconds")

                    print("\n4) Results")
                    print(f"Result:\nP-arrival: {p_utc}\nS-arrival: {s_utc}\nMagnitude: {mag_pred}\nPolarity: {pol} (Probability: {prob_pol:.3f})")
                    P_picks.append(p_pick)
                    S_picks.append(s_pick)
                    init_win.append(init_ref_pick)
                    picks_prob.append([prob_p, prob_s])
                    mags.append(mag_pred)
                    polarity_pred.append([pol, prob_pol])

                    # **Visualization**
                    plot_range = np.arange(
                        max(0, p_pick - int(2 * len_win * sampling / 3)),
                        min(len(prepro_wave_array), p_pick + len_win * sampling)
                    )
                    dat = prepro_wave_array[plot_range, :]

                    date_utc, _ = date_format_object(day=date, time=time_id)
                    starttime_str = date_utc.strftime("%Y-%m-%d %H:%M:%S")
                    
                    fig1 = plot_data(dat, station_id, date_utc, starttime_str, samples=plot_range,
                                     P_arr=p_pick, S_arr=s_pick, magnitude=mag_pred)
                    
                    fig2 = plot_dynapicker_output(dat[:,2], station_id, date_utc, starttime_str,
                                                  prob_p=prob_p, prob_s=prob_s,
                                                  P_arr=p_pick, S_arr=s_pick,
                                                  samples=plot_range, timepr0=init_ref_pick)
                    
                    fig3 = plot_polarcap_output(prepro_wave_array[:][:, 2], station_id,
                                                date_utc, p_pick, y_pred=[pol, prob_pol])

                    # **Save plots and Results to CSV if required**
                    if save_result:
                        os.makedirs(dirout_specific, exist_ok=True)
            
                        print(f"\nSaving results in {dirout_specific}{out_file}.csv")
                        pred = (p_pick, s_pick, mag_pred, pol, prob_pol)
                        save_results(dirout_specific, f"{out_file}.csv", prepro_stream[0].stats.network,
                                     prepro_stream[0].stats.station, prepro_stream[0].stats.location,
                                     start_time, prepro_stream[0].stats.endtime, pred)
                        
                        print(f"\nSaving plots in {dirout_specific}")                        
                        fig1.savefig(os.path.join(dirout_specific,f"{idet}_Pick_{p_utc}.pdf"), bbox_inches='tight')
                        fig2.savefig(os.path.join(dirout_specific,f"{idet}_Probpick_{p_utc}.pdf"), bbox_inches='tight')
                        fig3.savefig(os.path.join(dirout_specific,f"{idet}_PolarP_{p_utc}.pdf"), bbox_inches='tight')
                        

                    # Move to the next event
                    next_indx = id1_0f + len_win_id
                    if win_id1[-1] > next_indx:
                        icount = win_id1.index(next(filter(lambda x: x >= next_indx, win_id1)))
                    elif win_id1[-1] <= next_indx or len_id1 - icount <= detection_windows:
                        print("Analysis completed. No more events detected.")
                        break
            else:
                t_evaluated = start_time + (id1_0f * shift / sampling) - 1
                print(f"No events detected till {t_evaluated}")
                icount = win_id1.index(id1_0f)

    print(f"Total time elapsed: {time.time() - itime1:.2f} seconds")

    outputs = {
        'Start_time': start_time,
        'P_picks': P_picks,
        'S_picks': S_picks,
        'init_window': init_win,
        'picks_prob': picks_prob,
        'magnitudes': mags,
        'polarities': polarity_pred
    }

    return outputs
    
# ____________________________________________________                  
# _____ Old versions ___________________________

def detect(raw_waveform, device, idx_list, leng_win, picker_num_shift, magnitudes, batch_size=4, fremin=1, 
                fremax=40, fo=5, fs=100, bandpass_filter_flag=True, path=''):
    
    start_sample = idx_list[0]*10
    data = raw_waveform[start_sample:start_sample+leng_win*100,:] 
    stream = make_stream_array(data)
    
    # phase picking
    model = load_dynapicker(path = path)
    prob_p, prob_s, pwave, swave = phase_picking(device, model, stream, bandpass_filter_flag, picker_num_shift, 
                                                 batch_size, fremin, fremax, fo, fs)

    p_pick = pwave[0] + start_sample
    s_pick = swave[0] + start_sample
    
    # magnitude estimation
    
    mag_pred = round(np.mean([m[1] for m in magnitudes]),1)
    
    # polarity estimation
    polarcap = PolarCAP(path)
    X = raw_waveform[int(p_pick)-32:int(p_pick) + 32, 2]
    predictions = polarcap.predict(X.reshape(1, X.shape[0], 1))
    return [p_pick, s_pick, mag_pred, predictions[0][0], predictions[0][1]]


def monitor1(wsp, network, station, location, channel, start_time, end_time, device,
             leng_win, detection_windows = 5, shift=10,
             picker_num_shift=1, model_path = '', save_result=False, path=None, file_name=None):
    
    ## making sure to remove any existing result file with same name
    if save_result:
        result_file = os.path.join(path, file_name)
        if os.path.exists(result_file):
            os.remove(result_file)
    
    print("Downloading data...")
    st = waveform_download(wsp, net=network, sta=station, loc=location, chan=channel, starttime=start_time, endtime=end_time)
    print(st)
    print("Pre-processing data...")
    resample_stream, raw_waveform = preprocessing(st)
    print(resample_stream)
    plot_waveform(raw_waveform)
    print("Monitoring...")
    
    #window_waveform, creime_output = data_windows(raw_waveform, shift=shift)
    window_waveform, creime_output = windows_for_creime(model_path, raw_waveform, shift=shift)
    
    win_id1 = [] # all windows labeled as earthquqke
    for i in range(len(creime_output)):
        if creime_output[i][0] == 1:
            win_id1 += [i]
#     print(win_id1)
    # window selection
    win_id2 = [] 
    i, len_win = 0, len(win_id1)
    
    P_picks, S_picks, mags, polarity_pred = [], [], [], []
    while i < len_win:
        if i+detection_windows <= len_win-1: 
            index_list = win_id1[i:i+detection_windows]
            flag, id = check_continuity(index_list)
#             print(id, index_list)
            if flag:
                win_id2 += [index_list]
                # prediction
                magnitudes = creime_output[index_list[0]:index_list[-1]+1]
#                 print(magnitudes)
                pred = detect(raw_waveform, device, index_list, leng_win, picker_num_shift, magnitudes, 
                              batch_size=4, fremin=1, fremax=40, fo=5, fs=100, bandpass_filter_flag=True,
                              path=model_path)
                
                print('Result:\nP-arrival time = {}\nS-arrival time = {}\nMagnitude = {}\nPolarity = {} (Probability = {:.3f})'.\
                      format(start_time + pred[0]/100, start_time + pred[1]/100, pred[2] if pred[2] is not None else 0, pred[3], pred[4]))
                
                P_picks.append(pred[0])
                S_picks.append(pred[1])
                mags.append(pred[2])
                polarity_pred.append([pred[3], pred[4]])
                
                # visualization
                dat = raw_waveform[pred[0] - 100: min(len(raw_waveform), pred[0] - 100 + leng_win * 100), :]
                plot_waveform(dat, times=range(pred[0] - 100, min(len(raw_waveform), pred[0] - 100 + leng_win * 100)), 
                                  P_arr=pred[0], S_arr=pred[1], magnitude=pred[2])
                plot_polarcap_data(raw_waveform[pred[0] - 32:pred[0] + 32][:,2], y_pred = [pred[3],pred[4]])
                if save_result:
                    save_results(path, file_name, network, station, location, start_time, end_time, pred)
                next_ind = id + win_id1.index(id) + int(leng_win*10)
                if len(np.array(win_id1)[np.array(win_id1) > next_ind]) == 0:
                    break
                i = win_id1.index(np.array(win_id1)[np.array(win_id1) > next_ind][0])
                print("No (more) events detected till {}".format(start_time + win_id1[i] * shift /100))
            else:
                i = win_id1.index(id)+1
                print("No (more) events detected till {}".format(start_time + win_id1[i] * shift /100))
                
        else: 
            break
                
    outputs = {
        'P_picks': P_picks, 
        'S_picks': S_picks, 
        'magnitudes': mags, 
        'polarities': polarity_pred
    }
        
    return outputs

def monitor2(path_to_stream, device, leng_win, format=None,
             shift = 10, picker_num_shift = 10, detection_windows = 5, model_path='',
             save_result = False, path = './', file_name = None):
    
    ## making sure to remove any existing result file with same name
    if save_result:
        result_file = os.path.join(path, file_name)
        if os.path.exists(result_file):
            os.remove(result_file)
    
    st = read(path_to_stream, format = format)
    start_time = st[0].times("utcdatetime")[0]  
    resample_stream, raw_waveform = preprocessing(st)

    window_waveform, creime_output = windows_for_creime(model_path, raw_waveform, shift=shift)
    
    dynapicker_output = phase_pick_func(raw_waveform, creime_output, device, picker_num_shift=picker_num_shift,
                                       fremin = 5, fremax = 40, bandpass_filter_flag = True, path=model_path)
    polarity_pred, wins =  polarity_classification(raw_waveform, dynapicker_output, path=model_path)

    win_id1 = [] # all windows labeled as earthquqke
    for i in range(len(creime_output)):
        if creime_output[i][0] == 1:
            win_id1 += [i]
#     print(win_id1)
    # window selection
    win_id2 = [] 
    i, len_win = 0, len(win_id1)
    
    P_picks, S_picks, mags, polarity_pred = [], [], [], []
    while i < len_win:
        if i+ detection_windows <= len_win-1: 
            index_list = win_id1[i:i+detection_windows]
            flag, id = check_continuity(index_list)
#             print(id, index_list)
            if flag:
                win_id2 += [index_list]
                # prediction
                magnitudes = creime_output[index_list[0]:index_list[-1]+1]
#                 print(magnitudes)
                pred = detect(raw_waveform, device, index_list, leng_win, picker_num_shift, magnitudes,
                                   batch_size=4, fremin=1, fremax=40, fo=5, fs=100, bandpass_filter_flag=True, path=model_path)
                
                print('Result:\nP-arrival time = {}\nS-arrival time = {}\nMagnitude = {}\nPolarity = {} (Probability = {:.3f})'.\
                      format(start_time + pred[0]/100, start_time + pred[1]/100, pred[2] if pred[2] is not None else 0, pred[3], pred[4]))
                
                P_picks.append(pred[0])
                S_picks.append(pred[1])
                mags.append(pred[2] if pred[2] is not None else 0)
                polarity_pred.append([pred[3], pred[4]])
                
                # visualization
                dat = raw_waveform[pred[0] - 100: min(len(raw_waveform), pred[0] - 100 + leng_win * 100), :]
                plot_waveform(dat, times=range(pred[0] - 100, min(len(raw_waveform), pred[0] - 100 + leng_win * 100)), 
                                  P_arr=pred[0], S_arr=pred[1], magnitude=pred[2])
                plot_polarcap_data(raw_waveform[pred[0] - 32:pred[0] + 32][:,2], y_pred = [pred[3],pred[4]])
                if save_result:
                    save_results(path, file_name, st[0].stats.network, st[0].stats.station, st[0].stats.location, st[0].stats.starttime, st[0].stats.endtime, pred)
                next_ind = id + win_id1.index(id) + int(leng_win*10)
                if len(np.array(win_id1)[np.array(win_id1) > next_ind]) == 0:
                    break
                i = win_id1.index(np.array(win_id1)[np.array(win_id1) > next_ind][0])
                print("No (more) events detected till {}".format(start_time + win_id1[i] * shift /100))
            else:
                i = win_id1.index(id)+1
                print("No (more) events detected till {}".format(start_time + win_id1[i] * shift /100))
                
        else: 
            break
                
    outputs = {
        'P_picks': P_picks, 
        'S_picks': S_picks, 
        'magnitudes': mags, 
        'polarities': polarity_pred
    }
        
    return outputs


def phase_pick_func(prepro_wave_array, creime_output, device, leng_win=30, picker_num_shift=1, batch_size=4,
                    fremin=1, fremax=40, fo=5, fs=100, bandpass_filter_flag=True, path=''):
    win_num = len(creime_output)
    win_id1 = [] # all windows labeled as earthquqke
    for i in range(len(creime_output)):
        if creime_output[i][0] == 1:
            win_id1 += [i]
    win_id2 = []
    len_id1 = len(win_id1)
    i =  0
    while i<len_id1:
        if i+5 <= len_id1-1: 
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
    model = load_dynapicker(path = path)
    for idx_list in win_id2:
        p_pick = 0
        s_pick = 0
        start_sample = idx_list[0] *10
        data = prepro_wave_array[start_sample:start_sample + leng_win*100,:] 
        stream =  make_stream_array(data)
        prob_p, prob_s, pwave, swave = phase_picking(device, model, stream, bandpass_filter_flag, picker_num_shift, 
                                                     batch_size, fremin, fremax, fo, fs)
        p_pick = pwave[0] + start_sample
        s_pick = swave[0] + start_sample
        creime_rt = CREIME_RT(path)
        _, mag_pred = creime_rt.predict(np.array([prepro_wave_array[start_sample:min(len(prepro_wave_array),
                                        start_sample + min(6000,leng_win*100)),:]]))
        mag_pred = mag_pred[0][1]
        dynapicker_outputs.append([p_pick, s_pick, mag_pred])

        return dynapicker_outputs

# STA/LTA method
def classic_picking(trigger_type, trace, nsta, nlta, thr_on, thr_off, plotFlag):
    df = trace.stats.sampling_rate
    if trigger_type == 'classic_sta_lta':
        trigger_func = classic_sta_lta
    elif trigger_type == 'recursive_sta_lta':
        trigger_func = recursive_sta_lta
    else:
        raise NameError(
            """
            The trigger type must be either 'classic_sta_lta' or 'recursive_sta_lta'
            """)
    cft = trigger_func(trace.data, int(nsta * df), int(nlta * df))       
    if plotFlag:
        plot_trigger(trace, cft, thr_on, thr_off)
    return cft
