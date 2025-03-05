import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.defense import MettackGAT
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse

#pubmed dataset
#This one uses up too much memory, will not work yet
data = Dataset(root='/tmp/', name='pubmed', setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

features = features.cpu()
adj = adj.cpu()
labels = labels.cpu()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_attack(useGCN, attack_structure):

  # Setup Surrogate Model
  surrogate = GCN(nfeat=features.shape[1],
                  nclass=labels.max().item()+1,
                  nhid=16,
                  dropout=0.5,
                  with_relu=False,
                  with_bias=True,
                  weight_decay=5e-4,
                  device=device) if useGCN else MetattackGAT(nfeat=features.shape[1],
                                                             nhid=8,
                                                             nclass=labels.max().item() + 1,
                                                             heads=8,
                                                             dropout=0.5,
                                                             device=device)

  surrogate = surrogate.to(device)
  if useGCN:
    surrogate.fit(features, adj, labels, idx_train)
  else:
    surrogate.fit(features, adj, labels, idx_train, verbose=True)

  model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=attack_structure, attack_features=(not attack_structure), device=device)
  model = model.to(device)
  #perturbations = int(0.05 * (adj.sum() // 2))
  perturbations = 5
  model.attack(features, adj, labels, idx_train, idx_unlabeled,perturbations, ll_constraint=False)

  # save tracked data to google drive
  architecture = "GCN" if useGCN else "GAT"
  attack_type = "structure" if attack_structure else "features"
  csv_name = "attack_tracking_" + "pubmed_" + attack_type + "_" + architecture + ".csv"
  pickle_name = "attack_tracking_" + "pubmed_" + attack_type + "_" + architecture + "_full_data.pkl"
  model.save_tracking_data(csv_name, pickle_name)

  folder_path = '/home/yshen349/AttackTest/DeepRobust/mettack_results'
  os.makedirs(folder_path, exist_ok=True)

  csv_path = os.path.join(folder_path, csv_name)
  pickle_path = os.path.join(folder_path, pickle_name)

  model.save_tracking_data(csv_path, pickle_path)

  df = pd.read_csv(csv_path)
  print("\nTracking data from CSV:")
  print(df)

  with open(pickle_path, 'rb') as f:
    full_data = pickle.load(f)
  print("\nKeys in pickle file:", full_data.keys())

torch.cuda.empty_cache()
run_attack(True, True)
