import numpy as np 
import networkx as nx 
import scipy 
from scipy import sparse
import pickle 
import grinpy
from funcs import * 
import time 


start = time.time()

R =[20, 30, 40]
D =[]  
K = [1,2,3,4]
prunes = []

for r in R:
    print("Creating connected dominating set for {}kwh...".format(r))

    # Instantiate set of neighborhoods (for pruning, although could also be used for g_ckds)
    A = sparse.load_npz("data/Reachability_03_{}kwh_UNDIRECTED.npz".format(r))
    print("\tAdjacency matrix {} kWh loaded...".format(r))

    count = A.shape[0]
    neighborhood = {n: set(A[:,n].nonzero()[0]) for n in range(count)}
    print("\tNeighborhood created")
    degrees = {n: len(neighborhood[n]) for n in range(count)}
    print("\tDegrees created")

    for k in K:
        print("\tk = {}".format(k))
        # print("\tAdjacency matrix loaded...")

        print("\tRunning G-CDS for {} and {}...".format(r, k))
        d = G_kCDS_adj(A, degrees, neighborhood, k = k)
        _set = set(n for n,c in d.items() if c == 'red')
        
        size1 = len(_set)
        print("\tG-kCDS completed. Cardinality of D {}kwh, k = {}:\t{}".format(r, k, size1))

        print("\tPruning set...")
        _set = prune_gckds_adj(A, neighborhood, _set, degrees, k)
        size2 = len(_set)
        print("\tPruning completed. New size is {}".format(size2))

        prunes.append((size1, size2))
        D.append(_set)
    print("==================================================================")

with open('data/ConnDomSets.pickle', 'wb') as f:
    pickle.dump(D, f)

with open('data/ConnDomPrunes.pickle', 'wb') as f:
    pickle.dump(prunes, f)

categories = []
for r in R:
    for k in K:
        cat = (r,k)
        categories.append(cat)

with open('data/ConnDomSets_categories.pickle', 'wb') as f:
    pickle.dump(categories, f)

end = time.time()

print("Time elapsed:\t", end - start)
