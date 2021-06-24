from functools import partial
from scipy import sparse
import sys
from funcs import * 


# Load arguments from terminal command 
args = sys.argv
max_roadclass = int(args[1])

# Load graph
H = nx.read_gpickle('data/BaseGraph0{}_NOR_wagrades.pickle'.format(max_roadclass))

# Construct reachability graphs for each kWh of 20, 30 and 40
kwhs = [10,15,20,30,40]

for kw in kwhs:
    print("Constructing reachability graph for {} kwh. This process takes several hours...".format(kw))
    g = construct_reachability_graph(H, kw)
    g = scipy.sparse.csr_matrix(g)
    sparse.save_npz("data/Reachability_0{}_{}kwh.npz".format(max_roadclass, kw), g)
    B = create_undirected_matrix_from_adjacency_matrix(g, debug_faulty_edge=True)
 
    # Save undirected reachability graph as adjacency matrix
    sparse.save_npz("data/Reachability_0{}_{}kwh_UNDIRECTED.npz".format(max_roadclass, kw), B)