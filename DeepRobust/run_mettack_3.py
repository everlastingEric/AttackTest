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
data = Dataset(root=data_dir, name='pubmed', setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)

# Check if data is already on GPU before preprocessing
print(f"Before preprocessing - adj device: {adj.device if hasattr(adj, 'device') else 'Not a tensor'}")
print(f"Before preprocessing - features device: {features.device if hasattr(features, 'device') else 'Not a tensor'}")
print(f"Before preprocessing - labels device: {labels.device if hasattr(labels, 'device') else 'Not a tensor'}")

# Preprocess data
print("Preprocessing data...")
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

features = features.cpu()
adj = adj.cpu()
labels = labels.cpu()

# CRITICAL: Force device to be CUDA if available, regardless of preprocess function behavior
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Selected device: {device}")

# IMPORTANT: Check if data is already a tensor and on the correct device
print(f"After preprocessing - adj type: {type(adj)}")
print(f"After preprocessing - features type: {type(features)}")
print(f"After preprocessing - labels type: {type(labels)}")

# Convert to tensors if not already
if not isinstance(adj, torch.Tensor):
    adj = torch.tensor(adj, device=device)
else:
    adj = adj.to(device)
    
if not isinstance(features, torch.Tensor):
    features = torch.tensor(features, device=device)
else:
    features = features.to(device)
    
if not isinstance(labels, torch.Tensor):
    labels = torch.tensor(labels, device=device)
else:
    labels = labels.to(device)

# Verify data is on GPU
print(f"After moving to device - adj device: {adj.device}")
print(f"After moving to device - features device: {features.device}")
print(f"After moving to device - labels device: {labels.device}")

def run_attack(useGCN, attack_structure):
    global features, adj, labels
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

    
    # CRITICAL: Verify the model is on GPU
    surrogate = surrogate.to(device)
    print(f"Surrogate model device: {next(surrogate.parameters()).device}")
    
    print(f"Training surrogate model: {'GCN' if useGCN else 'GAT'}")
    if useGCN:
        surrogate.fit(features, adj, labels, idx_train)
    else:
        surrogate.fit(features, adj, labels, idx_train, verbose=True)
    
    print("\nInitializing Metattack model...")
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=attack_structure, attack_features=(not attack_structure), device=device)
    
    # CRITICAL: Verify the attack model is on GPU
    model = model.to(device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Attack model parameter {name} device: {param.device}")
    
    # Print GPU memory status
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Set perturbations
    perturbations = 5
    print(f"Performing attack with {perturbations} perturbations")
    
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)

    # CRITICAL: Force inputs to be on GPU right before calling attack
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
