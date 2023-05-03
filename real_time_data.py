from obspy import read
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.signal.filter import bandpass
from models import CREIME_RT
import numpy as np
client = Client(
   "IRIS"
)

def download_windows(net, sta, loc, chan, starttime, endtime,shift=10):
    starttime = UTCDateTime(starttime)
    endtime = UTCDateTime(endtime)
    myStream = client.get_waveforms(net, sta, loc, chan, starttime, endtime)
    print(myStream)
    myStream = myStream.resample(100)
    myStream = myStream.detrend()
    myStream.plot()
    X=np.transpose(bandpass(myStream, 1, 45, 100))
    
    X_test = []

    for win in range(500,len(X),shift):
        length = 500
        X_test.append(X[win-length:win])
        
    return X_test
        
def CREIME_RT_cont_outputs(net, sta, loc, chan, starttime, endtime,shift=10):
    creime_rt = CREIME_RT()
    
    X = download_windows(net, sta, loc, chan, starttime, endtime,shift)
    
    _, predictions = creime_rt.predict(X)
    
    return predictions