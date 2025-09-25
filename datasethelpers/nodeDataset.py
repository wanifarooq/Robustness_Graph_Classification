import os.path as osp
import hashlib
import torch
from torch_geometric.data import Data, InMemoryDataset

from utils import set_seed

class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, n_classes, n_nodes, n_prop, n_sample_p_c, transform=None, pre_transform=None):
        self.n_classes = n_classes
        self.n_nodes = n_nodes
        self.n_prop = n_prop
        self.n_sample_p_c = n_sample_p_c
        self.identifier = self._hash_args(root, n_classes, n_nodes, n_prop, n_sample_p_c)
        super(CustomGraphDataset, self).__init__(osp.join(root, self.identifier), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def _hash_args(self, *args):
        arg_str = '_'.join(map(str, args))
        return hashlib.md5(arg_str.encode()).hexdigest()

    def _has_cached_data(self):
        return osp.exists(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['custom_graph_dataset.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        for class_label in range(self.n_classes):
            mean = class_label+(class_label*0.6)
            std = 1.5  
            set_seed(mean)
            distribution = torch.distributions.Normal(mean, std)
            for _ in range(self.n_sample_p_c):  
                n_nodes = 0
                while n_nodes < 1:  
                    n_nodes = int(torch.distributions.Poisson(self.n_nodes).sample())
                x = distribution.sample((n_nodes, self.n_prop))
                edge_index = torch.randint(0, n_nodes, (2, n_nodes * 2))
                edge_index, _ = torch.unique(edge_index, dim=1, return_inverse=True)
                y = torch.tensor([class_label])
                data = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(data)
        set_seed()
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


# dataset = CustomGraphDataset('data/custom7',n_classes=3, n_nodes=7, n_prop=5, n_sample_p_c=1000)
# print(dataset)