import networkx as nx
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import numpy as np
import shapely
from shapely.geometry import Point, Polygon, LineString 
from shapely.ops import nearest_points
from functools import partial
import sys
from funcs import * 
from scipy import sparse
import pickle
import time 

# G = nx.read_gpickle('data/BaseGraph03_NOR_wagrades.pickle').to_undirected()

np.random.seed(25)

# R = [10, 15, 20]
R = [20, 30, 40]
D = []

sets = []

index = ["range", "k", "n", "seed"]
data = pd.DataFrame(columns = index)
# Run k-domination for each reachability graph and append results to D
for r in R:
    print(r)
    g = sparse.load_npz('data/Reachability_03_{}kwh_UNDIRECTED.npz'.format(r))
    g = nx.convert_matrix.from_scipy_sparse_matrix(g)

    # Run for each value of k
    k = [1,2,3,4]
    for i in k:
        # Run for 5 different random seeds
        # for s in range(5):
        print(i,r)
        start = time.time()
        _set = randomized_k_dominating(g, k = i)
        end = time.time() 
        print("Time elapsed:\t", end-start)
        setlen = len(_set)
        sets.append(_set)

        # Save current iteration of sets
        with open('data/DomSets_alt.pickle', 'wb') as f:
            pickle.dump(sets, f)

    del(g)

# Save collection of sets
with open('data/DomSets_alt.pickle', 'wb') as f:
    pickle.dump(sets, f)


# Retrieve set parameters and save object
set_param = []
for r in R:
    k = [1,2,3,4]
    for i in k:
        set_param.append([r,k])
with open('data/DomSets_param.pickle', 'wb') as f:
    pickle.dump(set_param, f)
