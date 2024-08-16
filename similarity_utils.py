from scipy.spatial import distance
import numpy as np
import math
import networkx as nx
import pandas as pd
from Bio import pairwise2
from sklearn.preprocessing import normalize


# ===================================
# Functions to Calculate Similarities 
# ===================================

# MiRNA Functional Similarity
# Pair-wise Best, Pairs Average  

# [1] Wang D, Wang J, Lu M, et al. Inferring the human microRNA functional similarity and functional network based on microRNA-associated diseases[J]. Bioinformatics, 2010, 26(13): 1644-1650.

# mi_i: the index of miRNA i
# mi_j: the index of miRNA j
# d_sem_sim: the semantic similarity matrix of diseases
# adj_mid: the adjacency matrix of miRNA-disease

# Attention! The index of miRNA functional similarity matrix equals to the miRNA index of adj_mid.

def PBPA(mi_i, mi_j, d_sem_sim, adj_mid):
    disease_set_i = adj_mid[mi_i] > 0
    disease_set_j = adj_mid[mi_j] > 0
    disease_sim_ij = d_sem_sim[disease_set_i][:, disease_set_j]
    ijshape = disease_sim_ij.shape
    # If a miRNA is not associated with any disease, than then similarity between this miRNA and any other miRNA cannot be calculated.
    if ijshape[0] == 0 or ijshape[1] == 0:
        return np.nan
    return (sum(np.max(disease_sim_ij, axis=0)) + sum(np.max(disease_sim_ij, axis=1))) / (ijshape[0] + ijshape[1])

# d_sem_sim: the semantic similarity matrix of diseases
# adj_mid: the adjacency matrix of miRNA-disease

# get miRNA functional similarity matrix
def get_mi_fuc_sim(d_sem_sim, adj_mid):
    mi_len = adj_mid.shape[0]
    # mi_fuc_sim = np.zeros((mi_len, mi_len))
    mi_fuc_sim = np.eye(mi_len)
    for i in range(mi_len):
        for j in range(i + 1, mi_len):
            mi_fuc_sim[i, j] = mi_fuc_sim[j, i] = PBPA(i, j, d_sem_sim, adj_mid)
    return mi_fuc_sim

# get miRNA sequence similarity
def get_mi_seq_sim(mi_df):
    mi_len = len(mi_df)
    mi_seq_sim = np.eye(mi_len)
    for i in range(mi_len):
        mi_i_seq = mi_df.loc[i]['Sequence']
        mi_seq_sim[i, i] = pairwise2.align.globalxx(mi_i_seq, mi_i_seq, score_only=True)
        for j in range(i + 1, mi_len):
            mi_j_seq = mi_df.loc[j]['Sequence']
            mi_seq_sim[i, j] = mi_seq_sim[j, i] = pairwise2.align.globalxx(mi_i_seq, mi_j_seq, score_only=True)
    return normalize(mi_seq_sim, norm='max')

# Disease Semantic Similarity

# [2] Wang J Z, Du Z, Payattakool R, et al. A new method to measure the semantic similarity of GO terms[J]. Bioinformatics, 2007, 23(10): 1274-1281.

# diseases_dag: the DAG of diseases
# d: disease ID
# w: semantic contribution factor (0.5 by default)

# get semantic contribution values (type 1)
def get_SV(diseases_dag, d, w=0.5):
    S = diseases_dag.subgraph(nx.descendants(diseases_dag, d) | {d})
    SV = dict()
    shortest_paths = nx.shortest_path(S, source=d)
    for x in shortest_paths:
        SV[x] = math.pow(w, (len(shortest_paths[x]) - 1))
    return SV

# diseases_dag: the DAG of diseases
# d: disease ID

# get semantic contribution values (type 2)
def get_SV2(diseases_dag, d):
    S = diseases_dag.subgraph(nx.descendants(diseases_dag, d) | {d})
    SV = dict()
    for x in S.nodes():
        # The number of DAGs including t / the number of diseases
        SV[x] = -1 * math.log((len(nx.ancestors(diseases_dag, x)) + 1) / len(diseases_dag.nodes()))
    return SV

# SV_i: the SV of disease i
# SV_j: the SV of disease j

def Wang(SV_i, SV_j):
    intersection_value= 0
    for disease in (set(SV_i.keys()) & set(SV_j.keys())):
        intersection_value = intersection_value + SV_i[disease] + SV_j[disease]
    # if intersection_value == 0:
    #     return 0
    # if sum(SV_i.values()) == 0 and sum(SV_j.values()) == 0:
    #     return np.nan
    return intersection_value / (sum(SV_i.values()) + sum(SV_j.values()))

# diseases_dag: the DAG of diseases
# diseases: the node list of diseases_dag
# w: semantic contribution factor (0.5 by default)

# Attention: the index of disease semantic similarity matrix equals to the node list of diseases_dag

# get disease semantic similarity matrix
def get_d_sem_sim(diseases_dag, type, diseases, w):
    
    SVs = dict()
    for d in diseases:
        if type == 1:
            SVs[d] = get_SV(diseases_dag, d, w)
        elif type == 2:
            SVs[d] = get_SV2(diseases_dag, d)

    d_len = len(diseases)
    # d_sem_sim = np.zeros((d_len, d_len))
    d_sem_sim = np.eye(d_len)
    for i in range(d_len):
        for j in range(i + 1, d_len):
            d_i = diseases[i]
            d_j = diseases[j]
            d_sem_sim[i, j] = d_sem_sim[j, i] = Wang(SVs[d_i], SVs[d_j])
    
    return d_sem_sim

# adj: the adjacency matrix of miRNA-disease or disease-miRNA
# node_i: the index of miRNA or disease i
# node_j: the index of miRNA or disease j
# multiplier: the multiplier of GIPK similarity

def GIPK_sim(adj, node_i, node_j, multiplier):
    result = np.linalg.norm(adj[node_i] - adj[node_j]) ** 2
    result = math.exp(-1 * multiplier * result)
    return result

# adj: the adjacency matrix of miRNA-disease or disease-miRNA
# gamma: a hyper-parameter (1 by default)

# get gaussian interaction profile kernel similarity matrix
def get_gipk_sim(adj, gamma):
    
    all_edu_dists = []
    for i in range(len(adj)):
        all_edu_dists.append(np.linalg.norm(adj[i]) ** 2)
    multiplier = gamma / (sum(all_edu_dists) / len(adj))
    
    node_len = adj.shape[0]
    # gipk_sim = np.zeros((node_len, node_len))
    gipk_sim = np.eye(node_len)
    for i in range(node_len):
        for j in range(i + 1, node_len):
            gipk_sim[i, j] = gipk_sim[j, i] = GIPK_sim(adj, i, j, multiplier)
    
    return gipk_sim

def get_diseases_dag(disease_id, paths):
    
    diseases_dag = nx.DiGraph()
    
    d_node_list = disease_id['disease_id'].values.tolist()
    
    disease_disease_df = pd.read_table(paths['disease_disease_df'])[['ID1', 'ID2']].drop_duplicates()
    edge_list = [(edge[0], edge[1]) for edge in disease_disease_df.values]
    
    diseases_dag.add_nodes_from(d_node_list)
    diseases_dag.add_edges_from(edge_list)
    
    # print(nx.is_directed_acyclic_graph(diseases_dag))

    return diseases_dag, d_node_list

def get_and_save_disease_similarities(diseases_dag, d_node_list, paths, settings):

    d_sem_sim = get_d_sem_sim(diseases_dag, 1, d_node_list, 0.5)
    print('Got disease semantic similarity matrix!')
    d_sem_sim2 = get_d_sem_sim(diseases_dag, 2, d_node_list, 0.5)
    print('Got disease semantic similarity matrix 2!')

    return d_sem_sim, d_sem_sim2

def get_and_save_other_similarities(d_sem_sim, d_sem_sim2, adj_mid, adj_dmi, paths, settings):
    
    # Here d_sem_sim2 is not used, maybe using it can lead to better result
    mi_fuc_sim = get_mi_fuc_sim(d_sem_sim, adj_mid)
    print('Got miRNA functional similarity matrix!')
    
    mi_gipk_sim = get_gipk_sim(adj_mid, 1)
    print('Got gaussian interaction profile kernel similarity matrix for miRNAs!')
    d_gipk_sim = get_gipk_sim(adj_dmi, 1)
    print('Got gaussian interaction profile kernel similarity matrix for diseases!')
    
    return mi_fuc_sim, mi_gipk_sim, d_gipk_sim

