import torch
import numpy as np
import os
from deeprobust.graph.defense import GCN
from deeprobust.graph.defense import MettackGAT
from deeprobust.graph.global_attack import Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import pandas as pd
import pickle

# Extensive diagnostic information
print("=" * 50)
print("CUDA DIAGNOSTICS")
print("=" * 50)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA_VISIBLE_DEVICES env var: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device index: {torch.cuda.current_device()}")
    print(f"Current CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("CUDA is not available. Check your PyTorch installation or GPU drivers.")
print("=" * 50)

# Set the data directory
data_dir = '/tmp/'
os.makedirs(data_dir, exist_ok=True)

# Output directory
output_dir = '/home/yshen349/AttackTest/DeepRobust/mettack_results'
os.makedirs(output_dir, exist_ok=True)

# Load pubmed dataset
print("Loading pubmed dataset...")
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
    print("\nInitializing surrogate model...")
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
    
    # Create filenames
    architecture = "GCN" if useGCN else "GAT"
    attack_type = "structure" if attack_structure else "features"
    csv_name = f"attack_tracking_pubmed_{attack_type}_{architecture}.csv"
    pickle_name = f"attack_tracking_pubmed_{attack_type}_{architecture}_full_data.pkl"
    
    # Set full paths
    csv_path = os.path.join(output_dir, csv_name)
    pickle_path = os.path.join(output_dir, pickle_name)
    
    print(f"Saving tracking data to {csv_path}")
    
    try:
        # Save tracking data
        model.save_tracking_data(csv_path, pickle_path)
        
        # Verify CSV was created
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"\nSuccessfully saved and loaded CSV file at {csv_path}")
            print("\nTracking data from CSV (first 5 rows):")
            print(df.head())
            print(f"CSV file shape: {df.shape}")
        else:
            print(f"Warning: CSV file was not created at {csv_path}")
        
        # Verify pickle was created
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                full_data = pickle.load(f)
            print(f"\nSuccessfully saved and loaded pickle file at {pickle_path}")
            print("\nKeys in pickle file:", list(full_data.keys()))
        else:
            print(f"Warning: Pickle file was not created at {pickle_path}")
            
    except Exception as e:
        print(f"Error saving or loading tracking data: {str(e)}")

# Clear GPU cache before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("Cleared GPU cache")

# Run attack
print("\nStarting attack...")
run_attack(True, True)
print("Attack completed")
