import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets.gnn_benchmark_dataset import GNNBenchmarkDataset
from torch_geometric.transforms import NormalizeFeatures
from ogb.graphproppred import PygGraphPropPredDataset
from torchvision.datasets import MNIST,CIFAR100,CIFAR10

from datasethelpers.nodeDataset import CustomGraphDataset
#this is the implementation for the customized class for tudataset so to change the get item method

class controNodefreq(CustomGraphDataset):
    def __init__(self, root='data/customSep_5',n_classes=6, n_nodes=7, n_prop=5, n_sample_p_c=1400):
        self.changed_indices = torch.Tensor()
        self.class_index = []
        self.dict_index ={}
        self.train_indices = torch.Tensor()

        super(controNodefreq, self).__init__(root=root, n_classes=n_classes, n_nodes=n_nodes,
                                             n_prop=n_prop, n_sample_p_c=n_sample_p_c)

    def __getitem__(self, item):
        data= super().__getitem__(item)
        indexes = [item] if not isinstance(item, (list, torch.Tensor)) else item
        return data,indexes
    


class CustomizeData(TUDataset):
    def __init__(self, root='data/ENZYMES', name='ENZYMES',attr=True ,transform = None):
        self.changed_indices = torch.Tensor()
        self.class_index = []
        self.dict_index ={}
        self.train_indices = torch.Tensor()

        super(CustomizeData, self).__init__(root=root, name=name, use_node_attr=attr, transform = transform)

    def __getitem__(self, item):
        data= super().__getitem__(item)
        indexes = [item] if not isinstance(item, (list, torch.Tensor)) else item
        return data,indexes
    

class CnnMnist(MNIST):
    def __init__(self, root='data/Mnist', train=True, download=True, transform=NormalizeFeatures()):
        self.changed_indices = torch.Tensor()
        self.class_index = []
        self.dict_index ={}
        self.train_indices = torch.Tensor()
        
        super(CnnMnist, self).__init__(root=root,train=train, download=download, transform=transform )

    def __getitem__(self, item):
        data= super().__getitem__(item)
        indexes = [item] if not isinstance(item, (list, torch.Tensor)) else item
        return data,indexes
    

class CnnCifar(CIFAR100):
    def __init__(self, root='data/Cifar', train=True, download=True, transform=NormalizeFeatures()):
        self.changed_indices = torch.Tensor()
        self.class_index = []
        self.dict_index ={}
        self.train_indices = torch.Tensor()
        
        super(CnnCifar, self).__init__(root=root,train=train, download=download, transform=transform )

    def __getitem__(self, item):
        data= super().__getitem__(item)
        indexes = [item] if not isinstance(item, (list, torch.Tensor)) else item
        return data,indexes

class CnnCifar10(CIFAR10):
    def __init__(self, root='data/Cifar10', train=True, download=True, transform=NormalizeFeatures()):
        self.changed_indices = torch.Tensor()
        self.class_index = []
        self.dict_index ={}
        self.train_indices = torch.Tensor()
        
        super(CnnCifar10, self).__init__(root=root,train=train, download=download, transform=transform )

    def __getitem__(self, item):
        data= super().__getitem__(item)
        indexes = [item] if not isinstance(item, (list, torch.Tensor)) else item
        return data,indexes


    
class CustomizeDataMnist(GNNBenchmarkDataset):
    def __init__(self, root='data/MNIST', name='MNIST',split='train'):
        self.changed_indices = torch.Tensor()
        self.class_index = []
        self.dict_index ={}
        self.train_indices = torch.Tensor()

        super(CustomizeDataMnist, self).__init__(root=root, name=name, split=split)

    def __getitem__(self, item):
        data= super().__getitem__(item)
        indexes = [item] if not isinstance(item, (list, torch.Tensor)) else item
        return data,indexes
#this is the custom class to get the data from ogb datasets and i have overrided the getItem function
class Ogbdataset(PygGraphPropPredDataset):
    def __init__(self,name='ogbg-ppa', transform=NormalizeFeatures()):
        self.changed_indices = torch.Tensor()
        self.class_index = []
        self.dict_index ={}
        self.train_indices = torch.Tensor()

        super(Ogbdataset, self).__init__(name=name, transform=transform)

    def __getitem__(self, item):
        data= super().__getitem__(item)
        indexes = [item] if not isinstance(item, (list, torch.Tensor)) else item
        return data,indexes


def get_random_ids (tensor,percent):
    # Step 1: Determine the total number of elements
    total_elements = tensor.numel()

    # Step 2: Calculate 10 percent of the total number of elements
    num_values_to_select = int(total_elements *percent)

    # Step 3: Generate random indices
    random_indices = torch.randperm(total_elements)[:num_values_to_select]
    random_values = tensor.view(-1)[random_indices]
    return random_values

def select_random_values(tensor,percent,numClass,index):
    # Step 1: Determine the total number of elements
    total_elements = tensor.numel()

    # Step 2: Calculate 10 percent of the total number of elements
    num_values_to_select = int(total_elements *percent)

    # Step 3: Generate random indices
    random_indices = torch.randperm(total_elements)[:num_values_to_select]

    # Step 4: Retrieve the values at the randomly selected indices
    random_values = tensor.view(-1)[random_indices]

    #generate the random labels for the noise creation for particular percentage of the samples and that can be decided

    # noise = torch.randint(low=0, high=numClass, size=(num_values_to_select,))

    noise = torch.Tensor()

    while len(noise) < num_values_to_select:

        random_value = torch.randint(low=0, high= numClass, size=(1,))
        if random_value != index:
            noise= torch.cat((noise,random_value))
    noise = noise.to(torch.long)
    return random_values, noise
