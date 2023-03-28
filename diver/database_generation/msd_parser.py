import h5py
import os
import math
import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from typing import Tuple


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


def compute_pitch_network_stats(pitch_array: np.array) -> Tuple[float, float, float]:
    # Create the pitch network
    pitch_network, pitch_codewords = create_pitch_network(pitch_array)
    # Compute the average degree of the nodes
    nodes = np.unique(pitch_codewords)
    if len(nodes) > 1:
        average_degree = compute_average_graph_degree(pitch_network, nodes)
    else:
        average_degree = 0
    # Compute the entropy of the graph
    pitch_graph = nx.from_numpy_matrix(pitch_network)
    graph_entropy = shannon_entropy(pitch_graph)
    average_clustering = nx.average_clustering(pitch_graph, nodes)

    return average_degree, graph_entropy, average_clustering


def create_pitch_network(pitch_array: np.array,
                         threshold: float = 0.5) -> Tuple[csr_matrix, np.array]:
    # Compute the codewords
    pitch_codewords = np.matmul(np.array(pitch_array > threshold, dtype=int),
                                np.array([2**i for i in range(12)], dtype=int))
    # Create the graph
    pitch_graph = np.zeros((2**12, 2**12), dtype=int)
    for i in range(1, len(pitch_codewords)):
        source = pitch_codewords[i - 1]
        dest = pitch_codewords[i]
        pitch_graph[source][dest] = 1

    return csr_matrix(pitch_graph), pitch_codewords


def compute_average_graph_degree(graph: csr_matrix, nodes: float) -> float:
    dist_matr, _ = dijkstra(csgraph=graph,
                            directed=True,
                            indices=nodes,
                            return_predecessors=True)
    all_degrees = dist_matr.flatten()
    all_degrees = all_degrees[np.isinf(all_degrees) == False]
    all_degrees = all_degrees[all_degrees > 0]
    return np.mean(all_degrees)
    

# Reference: https://stackoverflow.com/questions/70858169/networkx-entropy-of-subgraphs-generated-from-detected-communities
def degree_distribution(graph: nx.graph.Graph) -> Tuple[float, float]:
    v_k = dict(graph.degree())
    v_k = list(v_k.values()) # we get only the degree values
    max_k = np.max(v_k)
    k_values= np.arange(0, max_k+1) # possible values of k
    p_k = np.zeros(max_k + 1) # P(k)
    for k in v_k:
        p_k[k] = p_k[k] + 1
    p_k = p_k / sum(p_k) # the sum of the elements of P(k) must to be equal to one
    
    return k_values, p_k


def shannon_entropy(graph: nx.graph.Graph) -> float:
    k, p_k = degree_distribution(graph)
    h = 0
    for p in p_k:
        if(p > 0):
            h = h - p * math.log(p, 2)
    return h
