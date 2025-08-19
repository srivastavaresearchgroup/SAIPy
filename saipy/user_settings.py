from datetime import datetime

#REQUIRED for seismogram data
def set_filename(station_name, date_key, file_format, channel=''):
    """
    User-defined function to generate the full filename.

    You can customize this function to match the file naming convention
    used in your dataset. The returned string should be the complete name
    to the data file you want to process.

    This function is called by utils/visualization.py and by load_streams in data/realdata.py

    Parameters:
        station_name (str): Station code (e.g., 'PK15').
        date_key (str): Date and time identifier (e.g., '20201009_150000').
        file_format (str): File extension including the dot (e.g., '.mseed').
        channel (str, optional): Channel code (e.g., 'E', 'N', or 'Z').
                              Empty in case of there is not no channel
                              in the name (the 3 channels in the same file). 
    Returns:
        str: Full name to the target data file.
    """
    
    filename = f"{station_name}.PR.EH{channel}..{date_key}{file_format}"
    
    #filename = f"GE.c0{station_name}..{channel}.D.{date_key}{file_format}"
    #filename = f"{station_name}{file_format}"
    
    return filename


#REQUIRED for seismogram data
def date_format_object(day_time=None, day=None, time=None):
    """
    You can customize this function if your date format in the file names 
    is different from the default format '%Y%m%d_%H%M%S' or Julian format '%Y.%j.%H%M%S'.

    User-defined function to convert a string representing a date and time
    into a Python datetime object.

    This function is called by utils/visualization.py

    Parameters:
        day_time (str, optional): Full date and time as a single string 
            (e.g., '20230516_143015' or '2020.129.050000').
        day (str, optional): Date part only (e.g., '20230516' or '2020.129').
        time (str, optional): Time part only (e.g., '050000').

    Returns:
        datetime: A Python datetime object representing the input string.
    """
    if day_time is None:
        day_time = f"{day}_{time}"  # Use "_" 
        # day_time = f"{day}.{time}"  # Use "." 

    date = datetime.strptime(day_time, '%Y%m%d_%H%M%S')
    # date = datetime.strptime(day_time, "%Y.%j.%H%M%S")  # For Julian date format

    return date, day_time


#OPTIONAL
import h5py
from obspy import UTCDateTime, Trace, Stream

def read_non_seismic_format(path_file, station, day_time):
    """
    You can customize this function only if your data do not have
    a known format supported by ObsPy (see known formats in the table:
    https://docs.obspy.org/master/packages/autogen/obspy.core.stream.read.html).

    This function is called by load_streams in data/realdata.py
    
    This example is for .hdf5 format
    """

    fid = h5py.File(path_file, 'r')
    data = fid['seismic_data'][...] # depends on your data
    date, _ = date_format_object(day_time=day_time)
    start_time = UTCDateTime(date) # depends on your data
    station_stream = Stream()
    for component in ["E", "N", "Z"]:
        channel = "EH" + component
        tr = Trace(header={'network': 'NET', 'station': station,
                           'channel': channel, 'sampling_rate':100,
                           'starttime':start_time})
        station_stream.append(tr)
        
    station_stream[0].data = data[0] #E
    station_stream[1].data = data[1] #N
    station_stream[2].data = data[2] #Z
    
    return station_stream
    