import os
import numpy as np
from random import randint
from obspy import UTCDateTime, read, Stream
from obspy.clients.fdsn import Client
from obspy.signal.filter import bandpass

from saipy.user_settings import set_filename, read_non_seismic_format

#_____________________________________________
def waveform_download(wsp, net, sta, loc, chan, starttime, endtime):
    '''
    parameters:
    wsp: Web Service Provider, gg. IRIS
    net: Network code, e.g. 'IU'
    sta: Station code, e.g. 'ANMO'
    loc: Location code, e.g. '00'
    chan: Channel code, e.g. 'BHZ'
    startime: (UTCDateTime, optional) – Start date and time
    endtime (UTCDateTime, optional) – End date and time.
    '''
    client = Client(wsp)
    starttime = UTCDateTime(starttime)
    endtime = UTCDateTime(endtime)
    stream = client.get_waveforms(net, sta, loc, chan, starttime, endtime)
    stream.merge(fill_value='interpolate')
    return stream

def load_streams(dirstat, station, day_time, format, chn=''):
    """
    Load obspy streams in 3 channels for the current station

    Parameters:
        dirstat      (str)  : Base directory for the files.
        station      (list) : Station ID.
        day_time     (dict) : date and start time as in the file name.
        format       (str)  : File format extension.
        chn  (str, optional): Channel code (e.g., 'E', 'N', or 'Z').
                              Empty in case the 3 channels are in the same file.
    Return:
        (station_stream): ObsPy Stream object, with 3 channels.
    """
        
    station_stream = Stream()  # Initialize an empty Stream for this station
    has_error = False  # Track errors for this station at this time step
    if not chn:
        chn = '_'
    for c in chn:
        filename = set_filename(station, day_time, format, channel=c)
        path_file = os.path.join(dirstat,filename)
        if os.path.exists(path_file):
            try:    
                station_stream += read(path_file)
            except Exception as e:
                try:
                    # if the data has non-seismic format (e.g. .hdf5)
                    station_stream = read_non_seismic_format(path_file, station, day_time)
                except Exception as e:
                    print(f"! Warning: {station}: Could not read {path_file}: {e}")
                    has_error = True
                    break  # Stop reading this station for this time step
        else:
            print(f"! Warning: {station}:  No such file {path_file}")
            has_error = True

    if has_error:
        print(f"! Skipping station {station} for day,time: {day_time}")
    
    return station_stream  # Return the station's Stream for this time step
    

def preprocessing(stream,resample_freq=100,freq_min=1,freq_max=45):

    for tr in stream:
        if tr.stats.sampling_rate != 100:
            print(f"Resampling {tr.id} from {tr.stats.sampling_rate} Hz to 100 Hz")
            tr.resample(sampling_rate=resample_freq)
    
    stream.detrend('linear')
    stream.detrend('demean')
    stream.filter('bandpass',freqmin=freq_min,freqmax=freq_max,corners=4,zerophase=True)
    preproc_stream = stream.copy()
    preproc_array = np.array([str_data.data for str_data in preproc_stream])
    preproc_X = np.transpose(preproc_array)
    return preproc_stream, preproc_X