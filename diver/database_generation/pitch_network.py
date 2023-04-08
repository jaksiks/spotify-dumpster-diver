import math
from typing import Tuple

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


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
    pitch_graph = nx.Graph(pitch_network)
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
