from random import randint
from obspy import read
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import numpy as np
from obspy.signal.filter import bandpass


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

def preprocessing(stream,resample_freq=100,freq_min=1,freq_max=45):
    resample_stream = stream.resample(sampling_rate=resample_freq)
    resample_stream = resample_stream.detrend()
    X = np.transpose(bandpass(resample_stream,freq_min,freq_max,resample_freq))
    
    return resample_stream, X