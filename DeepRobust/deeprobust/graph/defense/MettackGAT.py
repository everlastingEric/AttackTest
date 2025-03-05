#reformatted the GAT class to be compatible with Mettack

import torch
import torch.nn.functional as F
from deeprobust.graph.defense import GAT
import torch.optim as optim
from copy import deepcopy

class SimpleData:
    def __init__(self, x, edge_index, y, train_mask):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.train_mask = train_mask
        self.val_mask = ~train_mask

    def to(self, device):
        """Add device movement functionality"""
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.y = self.y.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        return self

class MetattackGAT(GAT):
    def __init__(self, nfeat, nhid, nclass, heads=8, output_heads=1, dropout=0.5, lr=0.01,
                 weight_decay=5e-4, with_bias=True, device=None):
        super(MetattackGAT, self).__init__(nfeat=nfeat, nhid=nhid, nclass=nclass,
                                          heads=heads, output_heads=output_heads,
                                          dropout=dropout, lr=lr, weight_decay=weight_decay,
                                          with_bias=with_bias, device=device)
        # Add required attributes for Metattack
        self.nclass = nclass
        self.nfeat = nfeat
        self.hidden_sizes = [nhid * heads]  # Store hidden layer sizes
        self.weight_decay = weight_decay
        self.with_bias = with_bias
        self.with_relu = True  # GAT uses ELU by default

    def forward(self, x, edge_index):
        """Modified forward method to work with both formats"""
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        """Modified early stopping training to work with our data format"""
        if verbose:
            print('=== training GAT model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask = self.data.train_mask
        val_mask = self.data.val_mask

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()

            # Forward pass with correct argument format
            output = self.forward(self.data.x, self.data.edge_index)

            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.data.x, self.data.edge_index)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val))
        self.load_state_dict(weights)

    def fit(self, features, adj, labels, idx_train, train_iters=200, initialize=True, verbose=False, **kwargs):
        """Modified fit method to handle both PyG and normal format data"""
        # Convert features and labels to tensor if they aren't already
        if not isinstance(features, torch.Tensor):
            features = torch.FloatTensor(features)
        if not isinstance(labels, torch.Tensor):
            labels = torch.LongTensor(labels)

        # Convert adj to edge_index format
        if isinstance(adj, torch.Tensor):
            if adj.is_sparse:
                edge_index = adj._indices()
            else:
                edge_index = torch.nonzero(adj).t().contiguous()
        else:
            # If adj is numpy array or scipy sparse matrix
            adj = torch.FloatTensor(adj.todense() if hasattr(adj, 'todense') else adj)
            edge_index = torch.nonzero(adj).t().contiguous()

        # Create train mask
        train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
        train_mask[idx_train] = True

        # Create data object with the SimpleData class
        data = SimpleData(
            x=features,
            edge_index=edge_index,
            y=labels,
            train_mask=train_mask
        )

        if initialize:
            self.initialize()

        self.data = data.to(self.device)
        self.train_with_early_stopping(train_iters, patience=100, verbose=verbose)

    def test(self):
        """Modified test method to work with our data format"""
        self.eval()
        test_mask = self.data.test_mask
        labels = self.data.y
        output = self.forward(self.data.x, self.data.edge_index)
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils.accuracy(output[test_mask], labels[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def predict(self):
        """Modified predict method to work with our data format"""
        self.eval()
        return self.forward(self.data.x, self.data.edge_index)
