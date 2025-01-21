#runs the metattack algorithm on various datasets and GNN architectures

import torch
import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack
from deeprobust.graph.utils import to_scipy
from scipy.sparse.csgraph import structural_rank

data = Dataset(root='/tmp/', name='cora', setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
                with_relu=False, device=device)
surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train)

model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, device=device)
model = model.to(device)
perturbations = int(0.05 * (adj.sum() // 2))
model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
modified_adj = to_scipy(model.modified_adj)

# compare sparsity of original and perturbed graph
print("original sparsity:")
o_edges = adj.count_nonzero()
o_max_edges = adj.get_shape()[0] * adj.get_shape()[1]
print(o_edges, "edges out of", o_max_edges, "possible edges,", "percent connected: ", (o_edges / o_max_edges) * 100, "%")
print("perturbed sparsity:")
p_edges = modified_adj.count_nonzero()
p_max_edges = modified_adj.get_shape()[0] * modified_adj.get_shape()[1]
print(p_edges, " edges out of ", p_max_edges, " possible edges,", "percent connected: ", (p_edges / p_max_edges) * 100, "%")

# compare rank of original and perturbed graph
o_rank = structural_rank(adj)
o_max_rank = adj.get_shape()[0]
print("original rank:", o_rank)
print ("full rank") if o_max_rank == o_rank else print("not full rank, max possible rank is:", o_max_rank)
p_rank = structural_rank(modified_adj)
p_max_rank = modified_adj.get_shape()[0]
print("perturbed rank:", p_rank)
print ("full rank") if p_max_rank == p_rank else print("not full rank, max possible rank is:", o_max_rank)

#compare degree distribution of original and perturbed graph
o_degrees = {}
for i in range(adj.get_shape()[0]):
    degree = adj.getrow(i).count_nonzero() + adj.getcol(i).count_nonzero()
    if degree in o_degrees:
        o_degrees[degree] = o_degrees[degree] + 1
    else:
        o_degrees[degree] = 1
p_degrees = {}
for i in range(modified_adj.get_shape()[0]):
    degree = modified_adj.getrow(i).count_nonzero() + modified_adj.getcol(i).count_nonzero()
    if degree in p_degrees:
        p_degrees[degree] = p_degrees[degree] + 1
    else:
        p_degrees[degree] = 1
print("original degree distribution:")
for d in sorted(o_degrees.keys()):
    print(d, ":", o_degrees[d], ",", (o_degrees[d] / o_edges) * 100, "% percent of entire graph")
print("perturbed degree distribution:")
for d in sorted(p_degrees.keys()):
    print(d, ":", p_degrees[d], ",", (p_degrees[d] / p_edges) * 100, "% percent of entire graph")
