import itertools
import math
import numpy as np
import os

#import datetime
from datetime import datetime
from datetime import timedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.dates import date2num

import obspy
from obspy import UTCDateTime, read

from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from saipy.data.realdata import preprocessing, load_streams
from saipy.user_settings import date_format_object

plt.rcParams.update({'font.size': 14, 'axes.labelpad':14})
plt.rcParams['font.family'] = ['Arial'] #, 'DejaVu Sans', 'sans-serif', 'Liberation Sans',

# -------------------------

# Multiple- Stations plots _____________________________________________________________________

def format_hhmmss(x, pos):
    return str(timedelta(seconds=x))

def plot_overview(all_detect_ev, date_key, clusters, path_dat, format, chn='',
                        p_arrival=True, s_arrival=False, filter=True, fmin=1, fmax=45):
    """
    Plot the seismic signals of stations belonging to the same cluster for a particular `date_key`.
    
    Parameters:
    - all_detect_ev (list): The list containing (date_key, detections) tuples.
    - date_key (str): The specific date in the appropriate format for the filename.
    - clusters (list): List of clusters where each cluster contains (station, index) pairs.
    - path_dat (str)  : Base directory for the files.
    - chn (str): Format of vertical channel in file name (e.g. 'Z')
    - format   (str)  : File format extension.
    - p_arrival, s_arrival (bool): Whether to plot the picks.
    - filter (bool): Whether to filter data (default: True).
    - fmin, fmax (int): If filter= True, minimum and maximum frequencies for the bp filter (default: 1-45 Hz).    
    Returns:
    - fig
    """
    
    # Find the detections corresponding to the given date_key
    #p, s, station, idx, start_time, mag
    detections_for_date = next(detections for key, detections in all_detect_ev if key == date_key)

    station_list = list({stat_i for sublist in clusters for stat_i, _ in sublist})
    
    ti = 0 #math.ceil(min(all_p) - t_before)
    tf = 3600 #len(math.ceil(max(all_p) + t_after)

    date_dt, _ = date_format_object(day_time=date_key)
    starttime_utc = UTCDateTime(date_dt) # for time array
    starttime_str = date_dt.strftime("%Y-%m-%d %H:%M:%S") # for titles

    fig, axes = plt.subplots((len(station_list)), 1, figsize=(9, 1.7 * len(station_list) + 2),
                                     sharex=True, gridspec_kw={'hspace': 0})
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.gcf().autofmt_xdate() # rotation of tick labels

    fig.suptitle(f"File date: {starttime_str}", fontsize=14)
    
    for i, ax in enumerate(axes):
         # Read waveform (Only Z channel)
        dirstat = os.path.join(path_dat,f"{station_list[i]}/")
        station_stream = load_streams(dirstat, station_list[i], date_key, format, chn=chn)
        
        if filter:
            preproc_stream, _ = preprocessing(station_stream,resample_freq=100,freq_min=fmin,freq_max=fmax)
            if len(station_stream) > 1:
                tr = preproc_stream[2]#Only Z channel
            else:
                tr = preproc_stream[0]#Only Z channel
        else:
            if len(station_stream) > 1:
                tr = station_stream[2] # no preprocesed
            else:
                tr = station_stream[0] # no preprocesed

        # Slice and make time array and p arrival time relative to initial
        # time of the day (00:00:00) in seconds
        tr_slice, start_dt, t0_datetimes = make_data_plot(tr, ti, tf, starttime_utc)
        #  Time array in seconds - Calculate time differences in seconds since 00:00:00
        time_array = [(dt - start_dt).total_seconds() for dt in t0_datetimes]
        
        # Plot the signal for this station
        ax.plot(time_array, tr_slice.data/max(abs(tr_slice.data)), color='black', linewidth=1)
        ax.text(0.05, 0.8, f"{station_list[i]} - EHZ", transform=ax.transAxes)
        ax.set_ylabel('Normalized\nAmplitude')
        ax.set_ylim(-1.05,1.05)
        ax.set_yticks(np.array([-0.5,0,0.5]))

    for i, cluster in enumerate(clusters):   
        for station, cluster_idx in cluster:          
            # Find the event info for this station using `station` and `index`
            event = [e for e in detections_for_date if e[2] == station and e[3] == cluster_idx]
            if event:
                # Retrieve event (p_pick, s_pick, station, index, start_time, mag)
                p_pick, s_pick = event[0][0:2]
                j = station_list.index(station)
                
                if p_arrival:
                    #  Get the time relative to the start of the P and S arrival in format hh:mm:ss
                    p_time = tr.stats.starttime.datetime + timedelta(seconds=p_pick) - start_dt
                    axes[j].axvline(p_time.total_seconds(), color='r', linestyle='--', label='P arrival', linewidth=0.5)
                if s_arrival:
                    s_time = tr.stats.starttime.datetime + timedelta(seconds=s_pick) - start_dt
                    axes[j].axvline(s_time.total_seconds(), color='b', linestyle='--', label='S arrival', linewidth=0.5)

    len_trace = tf-ti
    dt = (len_trace)/10 # maximum 10 ticks 
    reft = int(math.ceil(round(dt / 60, 2) * 60))
    ax.xaxis.set_major_formatter(FuncFormatter(format_hhmmss))
    ax.xaxis.set_major_locator(MultipleLocator(reft)) 
    ax.set_xlabel('Time [HH:MM:SS]')
    ax.set_xlim(time_array[0],time_array[-1]+1)
    plt.tight_layout() 
    plt.show()
    
    return fig


def make_data_plot(tr, ti, tf, starttime_utc):
    """
    Slice and make time array and p arrival time relative to initial time of the day (00:00:00) in seconds.
    
    Parameters:
    - tr: the stream in the vertical channel.
    - ti (int): initial time for the slice in seconds relative
                to the original start time of the file data.
    - tf (int): final time for the slice in seconds relative
                to the original start time of the file data.
    """ 
    # Slice
    slice_start =  starttime_utc+ ti #tr.stats.starttime
    slice_end = starttime_utc + tf
    tr_slice = tr.copy().trim(starttime=slice_start, endtime=slice_end)
        
    #  Get the time relative to the original start of the trace (not a datetime object)
    t0_datetimes = [slice_start + timedelta(seconds=t) for t in tr_slice.times()]
    t0_datetimes = [dt.datetime for dt in t0_datetimes]  # Convert each UTCDateTime to datetime
    
    # Convert slice start (UTCDateTime) to a datetime object for 00:00:00 comparison
    start_dt = starttime_utc.datetime.replace(hour=0, minute=0, second=0)  # 00:00:00

    return tr_slice, start_dt, t0_datetimes
    
def plot_multi_stations(detections_for_date, date_key,
                        cluster, i_cl,
                        path_dat, station_list, dir_stat_name, format, chn='',
                        filter=True, fmin=1, fmax=45,
                        t_before=10, t_after=15,
                        p_arrival=True, s_arrival=False, mag_legd=True):
    """
    Plot the seismic signals of stations belonging to the same cluster for a particular `date_key`.
    
    Parameters:
    - detections_for_date (list): Contains (p, s, station, idx, start_time, mag) tuples.
    - date_key (str): The specific date in the appropriate format for the filename.
    - cluster (list): Contains (station, index) pairs.
    - i_cl (int): Index of the cluster.
    - path_dat (str)  : Base directory for the files.
    - station_list (sr): Name list of stations.
    - dir_stat_name (str)  : Name of directory for the files of a particular station.
    - format   (str)  : File format extension.
    - chn (str): Format of vertical channel in file name (e.g. 'Z')
    - filter (bool): Whether to filter data (default: True).
    - fmin, fmax (int): If filter= True, minimum and maximum frequencies for the bp filter (default: 1-45 Hz).
    - t_before, t_after (int): Time window for plot. Time before and after the earliest P arrival.
    - p_arrival, s_arrival, mag_legd (bool): Whether to plot the picks and to text the Magnitude.

    Returns:
    - fig
    """

    date_dt, _ = date_format_object(day_time=date_key)
    starttime_utc = UTCDateTime(date_dt) # for time array
    starttime_str = date_dt.strftime("%Y-%m-%d %H:%M:%S") # for titles

    fig, axes = plt.subplots(len(cluster), 1, figsize=(9, 1.7 * len(cluster) +2),
                                 sharex=True, gridspec_kw={'hspace': 0})

    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.gcf().autofmt_xdate() # rotation of tick labels

    fig.suptitle(f"Event {i_cl+1} - File date: {starttime_str}", fontsize=14)
    
    all_p = []
    info_by_station = []
    for station, cluster_idx in cluster:          
        # Find the event info for this station using `station` and `index`
        event = [e for e in detections_for_date if e[2] == station and e[3] == cluster_idx]
        if event:
            # Retrieve event (p_pick, s_pick, station, index, start_time, mag)
            p_pick, s_pick, station_name, event_idx, start_time, mag = event[0]
            all_p.append(p_pick)
            info_by_station.append((station_name, p_pick, s_pick, mag))

    ti = math.ceil(min(all_p) - t_before)
    tf = math.ceil(max(all_p) + t_after)

    for ax, info_stat in zip(axes, info_by_station):
        station_name, p_pick, s_pick, mag = info_stat

        # Read waveform (Only Z channel)
        index = station_list.index(station_name)
        dirstatnm = dir_stat_name[index]
        dirstat = os.path.join(path_dat,f"{dirstatnm}/")
        station_stream = load_streams(dirstat, station_name, date_key, format, chn=chn)

        if filter:
            preproc_stream, _ = preprocessing(station_stream,resample_freq=100,freq_min=fmin,freq_max=fmax)
            if len(station_stream) > 1:
                tr = preproc_stream[2]#Only Z channel
            else:
                tr = preproc_stream[0]#Only Z channel
        else:
            if len(station_stream) > 1:
                tr = station_stream[2] # no preprocesed
            else:
                tr = station_stream[0] # no preprocesed

        # Slice and make time array and p arrival time relative to initial
        tr_slice, start_dt, t0_datetimes = make_data_plot(tr, ti, tf, starttime_utc)
        
        #  Time array in seconds - Calculate time differences in seconds since 00:00:00
        time_array = [(dt - start_dt).total_seconds() for dt in t0_datetimes]

        if mag_legd:                
            ax.plot(time_array, tr_slice.data/max(abs(tr_slice.data)), color='black', linewidth=1)
            ax.text(0.05, 0.8, f"{station_name} - EHZ (Mag: {mag:.1f})", transform=ax.transAxes)
        else:
            ax.plot(time_array, tr_slice.data/max(abs(tr_slice.data)), color='black', linewidth=1)
            ax.text(0.05, 0.8, f"{station_name} - EHZ", transform=ax.transAxes)                
        if p_arrival:
            #  Get the time relative to the start of the P and S arrival in format hh:mm:ss
            p_time = tr.stats.starttime.datetime + timedelta(seconds=p_pick) - start_dt
            ax.axvline(p_time.total_seconds(), color='r', linestyle='--', label='P arrival')
        if s_arrival:
            s_time = tr.stats.starttime.datetime + timedelta(seconds=s_pick) - start_dt
            ax.axvline(s_time.total_seconds(), color='b', linestyle='--', label='S arrival')
            
        ax.set_ylabel('Normalized\nAmplitude')
        ax.set_ylim(-1.05,1.05)
        ax.set_yticks(np.array([-0.5,0,0.5]))
        ax.legend(loc='lower left')

    len_trace = tf-ti
    dt = (len_trace)/10 # maximum 10 ticks 
    reft = int(math.ceil(round(dt / 60, 2) * 60))        
    ax.xaxis.set_major_formatter(FuncFormatter(format_hhmmss))
    ax.xaxis.set_major_locator(MultipleLocator(reft)) 
    ax.set_xlabel('Time [HH:MM:SS]')
    ax.set_xlim(time_array[0],time_array[-1])
    plt.tight_layout()            
    plt.show()

    return fig

# Single- Station plots (updated - 2nd version)  ___________________________________________________________________________________

def plot_data(data, station_id, date_utc, starttime_str, samples=None,
                  P_arr=None, S_arr=None, magnitude=None, loc_leg='lower left', samp_rate=100):
    
    fig, ax = plt.subplots(3,1,figsize = [7,5.25], sharex = True, gridspec_kw={'hspace': 0})
    
    maxdata = max(max(abs(data), key=max)) # for normalizing

    if samples is None:
        ax[2].set_xlabel('Samples')
        ax[2].set_xlim(0,len(data[:,0]))
    else:
        time = samples / samp_rate
        time_utc = [date_utc + timedelta(seconds=t) for t in time]

        if P_arr is not None:
            P_time = P_arr / samp_rate
            P_arr = date_utc + timedelta(seconds=P_time)
        if S_arr is not None:
            S_time = S_arr / samp_rate
            S_arr = date_utc + timedelta(seconds=S_time)

        ax[2].set_xlim(time_utc[0],time_utc[-1])
        ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        n_ticks = math.floor(len(samples)/800)
        tick_positions = time_utc[::n_ticks*samp_rate]
        ax[2].set_xticks(tick_positions)
        ax[2].set_xlabel('Time HH:MM:SS')
        plt.xticks(rotation=20)

    for i in range(3):
        if samples is None:
            ax[i].plot(data[:,i]/maxdata, 'k', linewidth=1)
        else:
            ax[i].plot(time_utc, data[:,i]/maxdata, 'k', linewidth=1)

        ax[i].set_ylim(-1.05,1.05)
        ax[i].set_yticks(np.array([-0.5,0,0.5]))
        
        if P_arr is not None:  
            ax[i].axvline(P_arr, 0.2,0.8, color = 'r', label = 'P arrival', linestyle='dashed')           
        if S_arr is not None:
            ax[i].axvline(S_arr, 0.2,0.8, color = 'b', label = 'S arrival')
            
        if magnitude is not None:
            fig.suptitle(f"File date: {starttime_str} - Station: {station_id}\nMagnitude = {magnitude : .1f}", y=0.93, fontsize=14)
        else:
            fig.suptitle(f"File date: {starttime_str} - Station: {station_id}",y=0.93, fontsize=14)
            
    ax[0].text(0.1, 0.8, 'E',transform=ax[0].transAxes)
    ax[1].text(0.1, 0.8, 'N',transform=ax[1].transAxes)
    ax[2].text(0.1, 0.8, 'Z',transform=ax[2].transAxes)
    
    ax[1].set_ylabel('Normalized Amplitude')
    if P_arr or S_arr is not None:
        ax[2].legend(loc=loc_leg)
        
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_dynapicker_output(data, station_id, date_utc, starttime_str, prob_p, prob_s, P_arr, S_arr, samples=None, timepr0=0, picker_num_shift=10, samp_rate=100, figure_size=(7,5.25)):
    
    fig, axes = plt.subplots(2, 1, figsize=figure_size, sharex=True, gridspec_kw={'hspace': 0})    
    fig.suptitle(f"File date: {starttime_str} - Station: {station_id}",y=0.93, fontsize=14)

    final_prob_P = convert_func(prob_p, picker_num_shift)
    final_prob_S = convert_func(prob_s, picker_num_shift)
    trace_time_prob  = list(map(lambda x: (x/1.0)+timepr0, list(final_prob_P.keys())))
    maxdata = max(abs(data)) # for normalizing

    if samples is None:
        axes[1].set_xlabel('Samples')
        axes[1].set_xlim(0,len(data))
    else:
        time = samples / samp_rate
        time_utc = [date_utc + timedelta(seconds=t) for t in time]
        time_prob = np.asarray(trace_time_prob) / samp_rate
        time_utc_prob = [date_utc + timedelta(seconds=t) for t in time_prob]

        P_time = P_arr / samp_rate
        P_arr = date_utc + timedelta(seconds=P_time)
        S_time = S_arr / samp_rate
        S_arr = date_utc + timedelta(seconds=S_time)
    
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        n_ticks = math.floor(len(samples)/800)
        tick_positions = time_utc[::n_ticks*samp_rate]
        axes[1].set_xticks(tick_positions)
        axes[1].set_xlim(time_utc[0],time_utc[-1])
        axes[1].set_xlabel('Time HH:MM:SS')
        plt.xticks(rotation=20)

    if samples is None:
        axes[0].plot(data/maxdata, 'k', linewidth=1)
        axes[1].plot(trace_time_prob, list(final_prob_P.values()), color='r', label='P', linestyle='dashed')
        axes[1].plot(trace_time_prob, list(final_prob_S.values()), color='b', label='S')
    else:
        axes[0].plot(time_utc, data/maxdata, 'k', linewidth=1)
        axes[1].plot(time_utc_prob, list(final_prob_P.values()), color='r', label='P', linestyle='dashed')
        axes[1].plot(time_utc_prob, list(final_prob_S.values()), color='b', label='S')
        
    axes[0].axvline(x = P_arr, linewidth=1, color = 'r', linestyle='dashed')
    axes[0].axvline(x = S_arr, linewidth=1, color = 'b')
    axes[0].text(0.1, 0.8, 'Z',transform=axes[0].transAxes)
    axes[0].set_ylabel('Normalized Amplitude')
    axes[0].set_ylim(-1.05,1.05)
    axes[0].set_yticks(np.array([-0.5,0,0.5]))
    
    axes[1].set_ylim(0,1.05)
    axes[1].set_ylabel('Probability')
    axes[1].legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_polarcap_output(X, station_id, date_utc, p_pick, y_true = None, y_pred = None, samp_rate=100):

    P_time = date_utc + timedelta(seconds= p_pick / samp_rate)
    
    ref_windw = 32
    X = X[p_pick - ref_windw:p_pick + ref_windw].astype(float)
    time = np.arange(- ref_windw,ref_windw)/samp_rate
    
    fig, ax = plt.subplots(figsize = [5,4])
    
    plt.plot(time, X/max(abs(X)), color = 'k')
    plt.text(0.1, 0.85, 'Z', transform=ax.transAxes)
    plt.axvline(0, ls = '--', lw = 1.5, color = 'red', label='P arrival')
    plt.xlabel('Time [seconds]')
    plt.ylabel('Normalized Amplitude')
    plt.ylim(-1.05,1.05)
    plt.xlim(time[0],time[-1])
    plt.legend(loc='lower left')
    
    if y_pred is not None and y_true is not None:
        plt.title(f"Station: {station_id} - P arrival: {P_time}\nTrue Polarity: {y_true[0].capitalize()}\nPredicted polarity: {y_pred[0]} - Probability = {y_pred[1] : .2f}", fontsize=14)
    
    elif y_pred is not None:        
        plt.title(f"Station: {station_id} - P arrival: {P_time}\nPredicted polarity: {y_pred[0]} - Probability = {y_pred[1] : .2f}", fontsize=14)
        
    elif y_true is not None:
        plt.title(f"Station: {station_id} - P arrival: {P_time}\nTrue Polarity: {y_true[0]}", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return fig

# _________________________________________________________________________________
# ____ Single- Station plots. Old (1rst version) _____________________________________________________________________________

def plot_waveform(data, times = None, P_arr = None, S_arr = None, magnitude = None):
#     if times is None:
#         times = np.linspace(0, len(data)/100, len(data))
    fig, ax = plt.subplots(3,1,figsize = [7,6], sharex = True)
    
    for i in range(3):
        if times is None:
            ax[i].plot(data[:,i], 'k')
        else:
            ax[i].plot(times, data[:,i], 'k')
        ax[i].set_ylim(np.min(data) * 1.05,np.max(data) * 1.05)
        
        if P_arr is not None:
            ax[i].axvline(P_arr, 0.2,0.8, color = 'r', label = 'P-pick')
        if S_arr is not None:
            ax[i].axvline(S_arr, 0.2,0.8, color = 'b', label = 'S-pick')
            ax[0].legend()
        if magnitude is not None:
            fig.suptitle("Magnitude = {:.1f}".format(magnitude))
    ax[0].text(0.9, 0.85, 'E',transform=ax[0].transAxes)
    ax[1].text(0.9, 0.85, 'N',transform=ax[1].transAxes)
    ax[2].text(0.9, 0.85, 'Z',transform=ax[2].transAxes)
    
    fig.text(0.5, 0.04, 'Time (samples)', ha='center')
    fig.text(0.04, 0.5, 'Amplitude (counts)', va='center', rotation='vertical')       
    plt.show()
    return fig

def plot_creime_data(X, y, y_pred = None):
    
    fig, ax = plt.subplots(4,1,figsize = [7,8], sharex = True)

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    p_arr = 512 - np.sum(np.array(y) > -0.5)
    
    for i in range(3):
        ax[i].plot(X[:,i], 'k')
        ax[i].set_ylim(np.min(X) * 1.05,np.max(X) * 1.05)
        if p_arr != 512:
            ax[i].axvline(p_arr, linestyle = 'dotted', color = 'b')
    
    ax[3].plot(y, 'b')
    ax[3].set_ylim(-4.5,8)
    if p_arr != 512:
        ax[3].vlines(p_arr, -4, 8, linestyle = 'dotted', color = 'b')
    ax[3].set_yticks(np.linspace(-4,6,6))
    ax[1].set_ylabel("Amplitude [counts]", fontsize = 14)
    ax[3].set_ylabel("Data label", fontsize = 14)
    ax[3].set_xlabel("Samples", fontsize = 14)

    if y_pred is not None:
        ax[3].plot(y_pred, 'g')
        p_pred = 512 - np.sum(np.array(y_pred) > -0.5)
        if p_pred != 512:
            ax[3].vlines(p_pred, -4, 8, linestyle = 'dotted', color = 'g')
        legend_elements = [Line2D([0], [0], color='b', label='Ground truth'),
               Line2D([0], [0], color='g', label='CREIME Prediction')]
        ax[3].legend(handles=legend_elements, loc='upper left')
        
    plt.show()
        
def plot_creime_rt_data(X, y, y_pred = None):
    
    fig, ax = plt.subplots(4,1,figsize = [7,8], sharex = True)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    p_arr = 6000 - np.sum(np.array(y) > -0.5)
    
    for i in range(3):
        ax[i].plot(X[:,i], 'k')
        ax[i].set_ylim(np.min(X) * 1.05,np.max(X) * 1.05)
        if p_arr != 6000:
            ax[i].axvline(p_arr, linestyle = 'dotted', color = 'b')    
#     ax[0].text(50, 1200, 'E component', fontsize = 13, bbox=props, family = 'serif')
    ax[3].plot(y, 'b')
    ax[3].set_ylim(-4.5,8)
    if p_arr != 6000:
        ax[3].vlines(p_arr, -4, 8, linestyle = 'dotted', color = 'b')
    ax[3].set_yticks(np.linspace(-4,6,6))
    ax[1].set_ylabel("Amplitude [counts]", fontsize = 14)
    ax[3].set_ylabel("Data label", fontsize = 14)
    ax[3].set_xlabel("Samples", fontsize = 14)

    if y_pred is not None:
        ax[3].plot(y_pred, 'g')
        legend_elements = [Line2D([0], [0], color='b', label='Ground truth'),
               Line2D([0], [0], color='g', label='CREIME Prediction')]
        ax[3].legend(handles=legend_elements, loc='upper left')
    plt.show()

    return fig

def norm(X):
    maxi = np.max(abs(X), axis = 1).astype(float)
    X_ret = X.copy().astype(float)
    for i in range(X_ret.shape[0]):
        X_ret[i] = X_ret[i] / maxi[i]
    return X_ret

def plot_polarcap_data(X, y_true = None, y_pred = None):
    fig = plt.figure(figsize = [5,4])
    plt.plot(norm(np.array([X]))[0], color = 'k')
    plt.axvline(32, ls = '--', lw = 1.5, color = 'red')
    plt.xlabel('Time samples', fontsize = 14)
    plt.ylabel('Normalised\nAmplitude', fontsize = 14)
    plt.ylim(-1,1)
    
    if y_pred is not None and y_true is not None:
        plt.title('True Polarity: {}\nPredicted polarity: {}\nProbability = {:.2f}'.format(y_true[0].capitalize(), y_pred[0], y_pred[1]))
    
    elif y_pred is not None:
        plt.title('Predicted polarity: {}\nProbability = {:.2f}'.format(y_pred[0], y_pred[1]))
        
    elif y_true is not None:
        plt.title('True Polarity: {}'.format(y_true[0])) 
    plt.show()
    return fig
    

def convert_func(prob, n_shift):
    '''window index to sample'''
    Prob={}
    for j in range(0, len(prob)):
        id = int(j * n_shift + 200)
        Prob[id] = prob[j]
        
    return Prob

def plot_dynapicker_stead(stream, dataset, prob_p, prob_s, picker_num_shift, figure_size, index, normFlag=False):
    
    final_prob_P = convert_func(prob_p, picker_num_shift)
    final_prob_S = convert_func(prob_s, picker_num_shift)

    trace_time_p  = list(map(lambda x: x/1.0, list(final_prob_P.keys())))
    trace_time_s  = list(map(lambda x: x/1.0, list(final_prob_P.keys())))
   
    trace_data = stream[index].data
    
    if normFlag:
        trace_data = trace_data /max(np.abs(trace_data))
    
    if index == 0:
        ch = 'E-W'
    elif index == 1:
        ch = 'N-S'
    else:
        ch = 'Vertical'
        
    ## plot
    fig = plt.figure(figsize=figure_size)
    plt.suptitle(ch, y=0.93)
    axes = fig.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.1})
        
    line1, = axes[0].plot(trace_data, color='k')
    axes[0].set_ylabel('Amplitude \n [counts]')
    ymin, ymax = axes[0].get_ylim()
    pl = axes[0].vlines(dataset.attrs['p_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
    sl = axes[0].vlines(dataset.attrs['s_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
    axes[0].legend(handles=[pl, sl], loc='upper right', borderaxespad=0.)  

    axes[1].plot(trace_time_p, list(final_prob_P.values()), 'C0', label='P_prob')
    axes[1].plot(trace_time_s, list(final_prob_S.values()), 'C1', label='S_prob')
    axes[1].set_ylim(0,1)
    axes[1].set_xlabel('Samples')
    axes[1].set_ylabel('Probability')
    axes[1].legend(loc='upper right')
    plt.show()
    
    return fig
    
def plot_dynapicker_instance(stream, row, prob_p, prob_s, picker_num_shift, index, figure_size):
    
    fig, axes = plt.subplots(2, 1, figsize=figure_size, sharex=True, 
                             gridspec_kw={'hspace' : 0.08, 'height_ratios': [1, 1]}
                            )
    
    final_prob_P = convert_func(prob_p, picker_num_shift)
    final_prob_S = convert_func(prob_s, picker_num_shift)

    trace_time_p  = list(map(lambda x: x/1.0, list(final_prob_P.keys())))
    trace_time_s  = list(map(lambda x: x/1.0, list(final_prob_S.keys())))
    
    if index ==0:
        ch = 'E-W'
    elif index == 1:
        ch = 'N-S'
    else:
        ch = 'Vertical'
        
    plt.suptitle(ch, y=0.93)
    
    for j in range(3):
        if index == j:
            axes[0].plot(stream[j].data, color='k')
            
    axes[0].set_ylabel('Amplitude \n [counts]')
    ymin, ymax = axes[0].get_ylim()
    pl = axes[0].vlines(row['trace_P_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
    sl = axes[0].vlines(row['trace_S_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
    axes[0].legend(handles=[pl, sl], loc='upper right', borderaxespad=0.)  

    axes[1].plot(trace_time_p, list(final_prob_P.values()), color='C0', label='P_prob')
    axes[1].plot(trace_time_s, list(final_prob_S.values()), color='C1', label='S_prob')
    axes[1].set_ylim(0,1)
    axes[1].set_xlabel('Samples')
    axes[1].set_ylabel('Probability')
    axes[1].legend(loc='best')
    plt.show()
    
    return fig

def plot_dynapicker_stream(stream, prob_p, prob_s, picker_num_shift, figure_size):
    
    fig, axes = plt.subplots(2, 1, figsize=figure_size, sharex=True, 
                             gridspec_kw={'hspace' : 0.08, 'height_ratios': [1, 1]}
                            )
    
    final_prob_P = convert_func(prob_p, picker_num_shift)
    final_prob_S = convert_func(prob_s, picker_num_shift)

    trace_time_p  = list(map(lambda x: x/1.0, list(final_prob_P.keys())))
    trace_time_s  = list(map(lambda x: x/1.0, list(final_prob_S.keys())))
    
    for j in range(3):
        axes[0].plot(stream[j].data, label=stream[j].stats.channel)
        
    axes[0].set_ylabel('Amplitude \n (counts)')
    axes[0].legend(loc = 'upper right')
    axes[1].plot(trace_time_p, list(final_prob_P.values()), color='C0', label='P_prob')
    axes[1].plot(trace_time_s, list(final_prob_S.values()), color='C1', label='S_prob')
    axes[1].set_ylim(0,1)
    axes[1].set_xlabel('Samples')
    axes[1].set_ylabel('Probability')
    axes[1].legend(loc='best')
    plt.show()
    
    return fig

def plot_dynapicker_train_history(train_loss, valid_loss, figure_size):
    '''
    Codes source from https://github.com/Bjarten/early-stopping-pytorch
    '''
    
    # visualize the loss as the network trained
    fig = plt.figure(figsize=figure_size)
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1 
    plt.axvline(minposs, linestyle='--', color='r', label='Early-stopping checkpoint')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, max(max(train_loss), max(valid_loss))) # consistent scale
    plt.xlim(0, len(train_loss) + 1) # consistent scale
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_dynapicker_confusionmatrix(y_true, y_pred, label_list, digits_num, figure_size, cmap=plt.cm.PuBuGn):
    '''plot comfusion matrix'''
    fig = plt.figure(figsize=figure_size)
    cm = confusion_matrix(y_true.tolist(), y_pred)
    tick_marks = np.arange(len(label_list))
    plt.xticks(tick_marks, label_list, rotation=45)
    plt.yticks(tick_marks, label_list)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.show()
    
    ## Number of digits for formatting output floating point values.
    metrics_report = metrics.classification_report(y_true, y_pred, target_names=label_list, digits=digits_num)
    
    return fig, metrics_report

def plot_precision_recall_curve(y_true, y_pred, y_pred_prob, label_list, figure_size):
    
    y_true=np.array(y_true)
    # Compute the ROC curve for each class separately
    precisions = dict() 
    recalls = dict()
    thresholds = dict()
    roc_auc = dict()
    y_prob = np.vstack(y_pred_prob)
    n_classes = y_prob.shape[1]  # Number of classes
    
    fig = plt.figure(figsize=figure_size)
    ax= plt.axes()
    marker_list = ['o', '^']
    cmap = mpl.cm.spring
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    for i in range(n_classes-1):
        precisions[i], recalls[i], thresholds[i] = precision_recall_curve(y_true, y_prob[:,i], pos_label=2)
        plt.plot(recalls[i], precisions[i], marker=marker_list[i], linestyle='None', label=label_list[i])

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="best")
    plt.show()
    
    return fig

def plot_roc_curve(y_true, y_pred, y_pred_prob, label_list, figure_size):
    
    y_true = np.array(y_true)
    # Compute the ROC curve for each class separately
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_prob = np.vstack(y_pred_prob)
    n_classes = y_prob.shape[1]  # Number of classes

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_prob[:,i], pos_label=2)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot the ROC curves for each class
    fig = plt.figure(figsize=figure_size)
    class_label = label_list
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='Class: {0} (AUC = {1:.2f})'.format(class_label[i], roc_auc[i]))

    # plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc="best")
    plt.show()
    
    return fig
