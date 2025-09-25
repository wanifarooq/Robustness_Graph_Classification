import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import dropout_adj
# from torch_geometric.transforms import RandomSubgraph
from torch_geometric.data import Data
from sklearn.utils import shuffle
import numpy as np
import random

from datasethelpers.customdataset import Ogbdataset

# Function to add random noise to features
def add_noise(x, p=0.1):
    noise = torch.randn_like(x)
    mask = torch.rand_like(x) < p
    return x + noise * mask



def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data
# Example dataset
dataset = Ogbdataset(name = "ogbg-ppa", transform = add_zeros)

# Select a random graph
data = dataset[random.randint(0, len(dataset))]

# Augmentation operations
augmentations = [
    "dropnode",
    "dropedge",
    "nodedropping",
    "featuremasking",
    "featureshuffling",
    "dropmessage",
    "subgraphmasking",
    "graphmixup",
    "graphtransformation",
    "gmixup",
    "labelpropagation",
    "graphsmote",
    "graphcrop"
]

# Perform augmentations
for augmentation in augmentations:
    if augmentation == "dropnode":
        num_nodes = data.num_nodes
        num_drop = int(num_nodes * 0.5)
        drop_nodes = random.sample(range(num_nodes), num_drop)
        data = data.clone()
        drop_mask = torch.tensor([x not in drop_nodes for x in range(num_nodes)], dtype=torch.bool)
        data.edge_index = data.edge_index[:, drop_mask[data.edge_index[0]] & drop_mask[data.edge_index[1]]]
        data.x = data.x[drop_mask]
        data.y = data.y[drop_mask]
    elif augmentation == "dropedge":
        data.edge_index, _ = dropout_adj(data.edge_index, p=0.5)
    elif augmentation == "nodedropping":
        num_nodes = data.num_nodes
        num_drop = int(num_nodes * 0.5)
        drop_nodes = random.sample(range(num_nodes), num_drop)
        data = data.clone()
        data.edge_index = data.edge_index.clone()
        data.edge_index = torch.tensor([[x for x in edge if x not in drop_nodes] for edge in data.edge_index], dtype=torch.long)
        data.x = torch.tensor([x for i, x in enumerate(data.x) if i not in drop_nodes], dtype=torch.float)
        data.y = data.y[torch.tensor([i for i in range(num_nodes) if i not in drop_nodes])]
    elif augmentation == "featuremasking":
        num_features = data.num_features
        num_mask = int(num_features * 0.5)
        mask_features = random.sample(range(num_features), num_mask)
        data.x[:, mask_features] = 0
    elif augmentation == "featureshuffling":
        data.x = shuffle(data.x, random_state=0)
    elif augmentation == "dropmessage":
        data.x = add_noise(data.x)
    elif augmentation == "subgraphmasking":
        data = RandomSubgraph(0.5)(data)
    elif augmentation == "graphmixup":
        mix_data = dataset[random.randint(0, len(dataset))]
        alpha = np.random.uniform(0, 1)
        data.x = alpha * data.x + (1 - alpha) * mix_data.x
        data.edge_index = mix_data.edge_index
        data.y = alpha * data.y + (1 - alpha) * mix_data.y
    elif augmentation == "graphtransformation":
        data.x = torch.sin(data.x)
    elif augmentation == "gmixup":
        mix_data = dataset[random.randint(0, len(dataset))]
        alpha = np.random.uniform(0, 1)
        data.x = alpha * data.x + (1 - alpha) * mix_data.x
        data.edge_index = mix_data.edge_index
        data.y = alpha * data.y + (1 - alpha) * mix_data.y
    elif augmentation == "labelpropagation":
        data = RandomLPA(3)(data)
    elif augmentation == "graphsmote":
        data = RandomGraphSMOTE()(data)
    elif augmentation == "graphcrop":
        num_nodes = data.num_nodes
        num_keep = int(num_nodes * 0.5)
        keep_nodes = random.sample(range(num_nodes), num_keep)
        data = data.clone()
        data.edge_index = torch.tensor([[x for x in edge if x in keep_nodes] for edge in data.edge_index], dtype=torch.long)
        data.x = torch.tensor([x for i, x in enumerate(data.x) if i in keep_nodes], dtype=torch.float)
        data.y = data.y[torch.tensor([i for i in range(num_nodes) if i in keep_nodes])]

print(data)
