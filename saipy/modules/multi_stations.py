import csv
import os
import networkx as nx

from collections import defaultdict, Counter
from obspy import UTCDateTime

def select_final_clusters(pre_clusters_resolved):
    def get_pattern(cluster):
        return tuple(sorted(d["station"] for d in cluster))
        
    patterns = [get_pattern(c) for c in pre_clusters_resolved]   
    most_common_pattern = Counter(patterns).most_common(1)[0][0]

    # Graph by shared detection
    det_to_clusters = defaultdict(set)
    for i, cluster in enumerate(pre_clusters_resolved):
        for d in cluster:
            det_to_clusters[d["id"]].add(i)

    G = nx.Graph()
    G.add_nodes_from(range(len(pre_clusters_resolved)))
    for clusters in det_to_clusters.values():
        lst = list(clusters)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                G.add_edge(lst[i], lst[j])

    final_clusters = []

    for component in nx.connected_components(G):
        component = list(component)
        if len(component) == 1:
            final_clusters.append([(d['station'], d['index']) for d in pre_clusters_resolved[component[0]]])
        else:
            best_score = -1
            best = None
            for idx in component:
                c = pre_clusters_resolved[idx]
                pattern = get_pattern(c)
                score = len(set(pattern).intersection(set(most_common_pattern)))
                if score > best_score:
                    best_score = score
                    best = c
            final_clusters.append([(d['station'], d['index']) for d in best])

    return final_clusters

def resolve_overlaps(pre_clusters):
    # Create an overlap graph by detection ID
    det_to_clusters = defaultdict(set)
    for i, cluster in enumerate(pre_clusters):
        for d in cluster:
            det_to_clusters[d['id']].add(i)

    # Build connectivity graph
    G = nx.Graph()
    G.add_nodes_from(range(len(pre_clusters)))
    for clusters in det_to_clusters.values():
        lst = list(clusters)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                G.add_edge(lst[i], lst[j])

    # Resolve by connected component
    resolved = []

    for component in nx.connected_components(G):
        subclusters = [pre_clusters[i] for i in component]
        merged = merge_if_possible(subclusters)
        resolved.extend(merged)

    return resolved

def merge_if_possible(clusters):
    # Attempt to merge clusters without repeated stations
    merged = []
    used = [False] * len(clusters)

    for i, c1 in enumerate(clusters):
        if used[i]:
            continue
        base = c1[:]
        stations = set(d['station'] for d in base)
        used[i] = True
        for j in range(i + 1, len(clusters)):
            if used[j]:
                continue
            c2 = clusters[j]
            stations_c2 = set(d['station'] for d in c2)
            if stations.isdisjoint(stations_c2):
                base += c2
                stations.update(stations_c2)
                used[j] = True
        merged.append(base)

    # The ones that couldn't be merged should also be returned
    for i, flag in enumerate(used):
        if not flag:
            merged.append(clusters[i])

    return merged

def build_pre_clusters(detections, delta_t, min_stat):
    detections = sorted(detections, key=lambda d: d['time'])
    clusters = []

    for d in detections:
        added = False
        for c in clusters:
            # Check time and stations
            if abs(d['time'] - c[0]['time']) <= delta_t and d['station'] not in [x['station'] for x in c]:
                c.append(d)
                added = True
                break
        if not added:
            clusters.append([d])
            
    clusters = [c for c in clusters if len(set(d['station'] for d in c)) >= min_stat]
    
    return clusters

# Time-difference & pattern clustering _________________________________________________________________
def cluster_detections(detections, delta_t, min_stat):
    """
    detections: list of dictionaries with keys:
        {
            'p': float,
            's': float,
            'station': str,
            'index': int,
            'time': float,
            'mag': float,
            'id': str   # unique per station + index
        }
    delta_t: maximum time between detections of the same cluster
    min_stat: minimum number of stations required per event.
    """
    pre_clusters = build_pre_clusters(detections, delta_t, min_stat)
    if not pre_clusters:
        print("Warning: No resolved clusters available for final event detection.")
        return []

    pre_clusters_resolved = resolve_overlaps(pre_clusters)
    if not pre_clusters_resolved:
        print("Warning: No resolved clusters available for final event detection.")
        return []

    final_clusters = select_final_clusters(pre_clusters_resolved)
    if not final_clusters:
        print("Warning: No resolved clusters available for final event detection.")
        return []
    else:
        return final_clusters

# Simple time-difference clustering_________________________________________________________________

def cluster_detections_dt(detections, delta_t, min_stat):
    """
    Clusters based on temporal proximity of detections in different stations.
    
    Parameters:
        detections: List of tuples (t, station, idx), where:
            - t: P arrival time
            - station: identifier of the station
            - idx: index of the event within its series of detections
        delta_t: Maximum time tolerance between P arrivals in the same cluster.
        min_stat: Minimum number of stations required per event.
    
    Returns:
        A list of clusters, where each cluster is a list of detections of the same earthquake by diferents stations
    """
    # Sort detections by time
    detections = sorted(detections, key=lambda x: x[0])
    
    clusters = []
    used = set()  # Tracks detections already assigned to a cluster
    
    for i, (tp_i, ts_i, station_i, idx_i, t0_i, mag_i) in enumerate(detections):
        if (tp_i, station_i, idx_i) in used:
            continue

        # Time in seconds considering the start time of the origin data
        t0_i = t0_i.timestamp
        tsec_i = t0_i + tp_i
        
        # Initialize a new cluster
        cluster = [(station_i, idx_i)]
        used.add((tp_i, station_i, idx_i))

        # Search for nearby detections in time
        for j in range(i + 1, len(detections)):
            tp_j, ts_j, station_j, idx_j , t0_j, mag_j= detections[j]
            
            # Time in seconds considering the start time of the origin data
            t0_j = t0_j.timestamp
            tsec_j = t0_j + tp_j
            
            # Stop searching if the neighbor is outside the time range
            if abs(tsec_j - tsec_i) > delta_t:
                break
            
            # Add to the cluster if it's from a different station
            if station_j != station_i and (tp_j, ts_j, station_j, idx_j, t0_j, mag_j) not in used:
                cluster.append((station_j, idx_j))
                used.add((tp_j, station_j, idx_j))
        
        # Add the cluster if it contains more than one event (optional)
        if len(cluster) >= min_stat:
            clusters.append(cluster)
    return clusters

# Save Events _________________________________________________________________

def save_clusters_to_csv(clusters, outputs_dict, filename='Events_output.csv', outpath='./'):
    """
    Saves information into a CSV file of Events detected in more than one station.
    
    Parameters:
    - clusters (list): List of clusters with tuples (station, index).
    - outputs_dict (dict): Dictionary containing results per stations from saipy monitor.
    - outpath (str): Directory path where the output CSV files will be saved (default: './').
    - filename (str): Output CSV file name (default: "Events_output.csv").
    """
    
    file_path = os.path.join(outpath, filename)
    file_exists = os.path.isfile(file_path)

   # Open in write mode ('w') to overwrite existing file
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)

        # Write CSV header
        writer.writerow(["Event ID", "Station", "Index", "P_picks", "S_picks", "P_UTC", "S_UTC", "Magnitude", "Polarity"])

        # Iterate over clusters with an ID counter
        for cluster_id, cluster in enumerate(clusters, start=1):
            for station, index in cluster:
                if station in outputs_dict:  # Ensure the station exists in outputs_dict
                    # Get reference time t0 from 'Start_time'
                    t0 = UTCDateTime(outputs_dict[station]['Start_time'])

                    # Extract data for the given index
                    p_pick = outputs_dict[station]['P_picks'][index]
                    s_pick = outputs_dict[station]['S_picks'][index]
                    magnitude = outputs_dict[station]['magnitudes'][index]
                    polarity = outputs_dict[station]['polarities'][index]

                    # Convert to seconds and get UTC time
                    p_utc = t0 + (p_pick / 100)  # P_picks in UTC
                    s_utc = t0 + (s_pick / 100)  # S_picks in UTC

                    # Write the row in CSV with the correct order
                    writer.writerow([cluster_id, station, index+1, p_pick, s_pick, p_utc, s_utc, magnitude, polarity])

    print(f"\nCSV file '{filename}' successfully updated.")