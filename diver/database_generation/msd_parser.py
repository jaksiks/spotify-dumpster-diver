import h5py
import os
import numpy as np
import pandas as pd

from diver.database_generation.pitch_network import compute_pitch_network_stats


def parse_msd_data_group(h5_file: h5py.File, field: str) -> dict:
    """
    Parse a data group of the H5 file

    :param h5_file: Current h5py file handle
    :param field: Field of interest in the h5 file
    :returns: The h5 file in dictionary form
    """
    data_group = h5_file[field]
    # Create a dictionary for each key with all data contained by the song
    data_dict = {}
    for key in data_group.keys():
        data_dict[key] = data_group[key][()]
    return data_dict


def msd_h5_to_df(filename: str) -> pd.DataFrame:
    """
    """
    with h5py.File(filename, "r") as f:
        msd_id = os.path.basename(f.filename).replace(".h5", "")
        analysis_dict = parse_msd_data_group(f, "analysis")
        metadata_dict = parse_msd_data_group(f, "metadata")
        musicbrainz_dict = parse_msd_data_group(f, "musicbrainz")
    
    temp_dict = {
        "msd_id": msd_id,
        "artist_id": metadata_dict["songs"][0]["artist_id"].decode("utf-8"),
        "artist_name": metadata_dict["songs"][0]["artist_name"].decode("utf-8"),
        "artist_familiarity": metadata_dict["songs"][0]["artist_familiarity"],
        "artist_hotttnesss": metadata_dict["songs"][0]["artist_hotttnesss"],
        "song_id": metadata_dict["songs"][0]["song_id"].decode("utf-8"),
        "song_title": metadata_dict["songs"][0]["title"].decode("utf-8"),
        "song_hotttnesss": metadata_dict["songs"][0]["song_hotttnesss"],
        "year": musicbrainz_dict["songs"][0]["year"],
        "loudness": analysis_dict["songs"][0]["loudness"],
        "energy": analysis_dict["songs"][0]["energy"],
        "danceability": analysis_dict["songs"][0]["danceability"],
        "tempo": analysis_dict["songs"][0]["tempo"]        
    }

    if np.isnan(temp_dict["song_hotttnesss"]):
        temp_dict["song_hotttnesss"] = 0

    # Get some summary features for the pitch arrays
    average_degree, graph_entropy, average_clustering = \
        compute_pitch_network_stats(analysis_dict["segments_pitches"])
    temp_dict["pitch_network_average_degree"] = average_degree
    temp_dict["pitch_network_entropy"] = graph_entropy
    temp_dict["pitch_network_mean_clustering_coeff"] = average_clustering

    # Get summary features of the timbre
    avg_timbres = np.mean(analysis_dict["segments_timbre"], axis=0)
    for i in range(len(avg_timbres)):
        temp_dict["timbre_{:02d}".format(i)] = avg_timbres[i]

    return pd.DataFrame.from_dict([temp_dict,])
