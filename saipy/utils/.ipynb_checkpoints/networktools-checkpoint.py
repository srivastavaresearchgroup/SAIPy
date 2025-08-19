import os
import time

from saipy.data.realdata import *
from saipy.modules.multi_stations import *
from saipy.user_settings import date_format_object
from saipy.utils.packagetools import monitor
from saipy.utils.visualizations import plot_multi_stations


def network_detect(output_by_station, delta_t=3.0, save_result=True,
                path_out='', path_dat='', channel='', format=''):
    """
    Clusters seismic detections across multiple stations within a time window.

    This function groups detections occurring within a specified time window (`delta_t`),
    and saves the clustered results and figures.

    Parameters:
    - output_by_station (dict): Dictionary where keys are date strings and values are 
      dictionaries containing seismic detection results per station.
    - delta_t (float, optional): Time window (in seconds) to cluster detections
        close in time by different stations (default: 3.0 seconds).
    - save_result (bool): Whether to save results (default: True).
    - if save_result = True:
        path_out (str)  : Directory path where the output CSV files and plots will be saved.
        path_dat (str)  : Base directory for the files.
        channel (str, optional): Vertical channel code (e.g., 'Z').
                              Empty in case of there is not no channel
                              in the name (the 3 channels in the same file).
        format   (str)  : File format extension.
        
    Returns:
    - clusters_for_plotting (list of dict): A flat list of detected clusters.
      Each item is a dictionary with:
        - 'date_key': the date associated with the cluster, in the format of the filename.
        - 'cluster': list of event tuples (p_pick, station, index, start_time, magnitude)
    
    - all_detect_ev (list of tuples): List of all detections organized by date_key.
      Format: [(date_key, [detection1, detection2, ...]), ...], where each detection is a tuple 
      (p_pick, station, index, start_time, magnitude).
    """

    clusters_for_plotting = []
    
    # Generate lists of detections for each date_key (date)
    all_detect_ev = [
        (date_key, [
            (p_pick/100, s_pick/100, station, index, data['Start_time'], mag)
            for station, data in stations.items()
            for index, (p_pick, s_pick, mag) in enumerate(zip(data['P_picks'], data['S_picks'], data['magnitudes']))
        ])
        for date_key, stations in output_by_station.items()
    ]
    
    # Process detections for each date_key
    for date_key, list_detections in all_detect_ev:

        dict_detections = [
            {
                'p': p,
                's': s,
                'station': station,
                'index': idx,
                'time': start_time + p,
                'mag': mag,
                'id': f'{station}_{idx}'
            }
            for p, s, station, idx, start_time, mag in list_detections
        ]
        # For time_difference & pattern clustering:
        clusters = cluster_detections(dict_detections, delta_t)
        
        # For Simple time_difference clustering:
        #clusters = cluster_detections_dt(list_detections, delta_t) 

        if clusters:
            print(f"\nResults for {date_key}: {len(clusters)} events")
            # Append clusters to clusters_for_plotting under the respective date_key
            clusters_for_plotting.append({
                'date_key': date_key,
                'cluster': clusters
            })
           
            path_datek_out = f"{path_out}Events_by_network/{date_key}/"

            # Save clustered detections to a CSV file
            if save_result:# and not path_exist: #
                # Ensure the output directory structure exists                
                os.makedirs(path_datek_out, exist_ok=True)
                
                save_clusters_to_csv(clusters, output_by_station[date_key], 
                                     filename=f"Events_{date_key}_output.csv", 
                                     outpath=path_datek_out)

            # Find the detections corresponding to the given date_key
            #p, s, station, idx, start_time, mag
            detections_for_date = next(detections for key, detections in all_detect_ev if key == date_key)

            # Iterate through the clusters and fetch the corresponding signals
            for i_cl, cluster in enumerate(clusters):
                fig_net = plot_multi_stations(detections_for_date, date_key, cluster, i_cl, path_dat, format, chn=channel)
                if save_result:
                    fig_net.savefig(f"{path_datek_out}/Event_{i_cl+1}_{date_key}.pdf", bbox_inches='tight')    
        else:
            print(f"\n*No events for {date_key}*")

    return clusters_for_plotting, all_detect_ev
    


def make_monitor_multistations(dirdata, dir_stat_name, station_list, channel, time_dict, format,
                    device, lw, dw, dirmodel, dirwork, dirout, save_result=True):

    """
    Calls `load_streams()` to retrieve seismic data for each station, 
    preprocesses the data using `preprocessing()`, and then performs event monitoring analysis
    using `monitor` on the preprocessed data.

    This function processes data for each station at each time step (defined by day and hour) and returns the
    results of the analysis grouped by day and hour.

    Parameters:
        dirdata      (str)  : Base directory for seismic data files.
        dir_stat_name (list) : List of name folders of each station.
        station_list (list) : List of station IDs to process.
        channel         (list) : List of channel names (['E', 'N', 'Z']).
                              Empty in case of there is not no channel
                              in the name (the 3 channels in the same file).
        time_dict    (dict) : Dictionary where the keys are the date (str) and values
                              are lists of start time (str) to process.
        format       (str)  : File format extension or suffix for seismic data files (e.g., '.mseed', '.sac').
        device       (object): The device used for processing, typically a trained model or computational resource.
        lw           (int)  : Length of the scanning time window in seconds.
        dw           (int)  : Number of consecutive positive detections to confirm an event.
        dirmodel     (str)  : Path to the model used for analysis.
        dirwork      (str)  : Main Directory for working.
        dirout       (str)  : Directory to save the processed results.
        save_result  (bool) : Whether to save results (default: True).

    Return:
        outputs_all: A dictionary of results for each station processed during the given day and start time.
              The dictionary is grouped by "day and start time" as keys and contains processed results for each station.
              
    Functions Called:
        - `load_streams()`: Loads seismic data for each station and time step (day, hour).
        - `preprocessing()`: Preprocesses the loaded seismic data (resampling, detrending, filtering).
        - `monitor`: Analyzes the preprocessed data for event monitoring.
    """

    outputs_all = {}  # Initialize the dictionary to store final outputs for each day-hour 

    for day, start_time in time_dict.items():  # Loop through all days in the time_dict
        for stime in start_time:  # Loop through all hours for a given day
            
            outputs_by_station = {}  # Initialize the dictionary to store outputs for each station within this day-hour 

            date, day_time = date_format_object(day=day, time=stime)
            starttime_str = date.strftime("%Y-%m-%d %H:%M:%S") # for titles
            
            for ist, station in enumerate(station_list):
                dirstat = os.path.join(dirdata,f"{dir_stat_name[ist]}/")
                 
                print(f"\n** Reading data: station {station}, date {starttime_str}... **")
                
                stream = load_streams(dirstat, station, day_time, format, chn=channel)

                if len(stream) < 3: # No data available
                    continue  # Skip processing

                print("\n** Preprocessing data... **")
                itime0 = time.time()
                prepro_stream, prepro_wave_array = preprocessing(stream)
                print(f"Time elapsed: {str(time.time() - itime0)} ")

                outputs = monitor(prepro_stream, prepro_wave_array, device,
                                  station, day, stime,
                                  len_win=lw, detection_windows=dw,
                                  model_path=dirmodel,
                                  save_result=save_result,
                                  outpath=dirout)

                outputs_by_station[station] = outputs  # Store results for each station

            # Store the results for this day-start time in the outputs_all dictionary
            outputs_all[day_time] = outputs_by_station

            print(f"\n** End of Single-station monitoring: station {station}, date {starttime_str}. *******************************\n")

    return outputs_all  # Return the processed results grouped by day and hour