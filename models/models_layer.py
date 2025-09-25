import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
#for now i have installed the cpu version of the dgl which i will be uninstalling later
#i have right now 12.1 cuda version and 2.1.0 torch vesion which is not currently supported by
#current version of gpu supported dgl now for testing pupose i have included that
import dgl
# from dgl.nn.pytorch.glob import SortPooling
from models.layers import GCN, HGPSLPool
from torch_geometric.nn import GCNConv,JumpingKnowledge
from torch_geometric.nn import GINConv,global_add_pool,GATv2Conv
# this one i have added just for to test the VPA
from torch_geometric.nn import global_add_pool,GATv2Conv
from torch_geometric.nn.conv import GINConv
# from VPA.src.tg_vpa.gin_conv import GINConv
from torch.nn import BatchNorm1d
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

class HGPSLModel(torch.nn.Module):
    def __init__(self, args):
        super(HGPSLModel, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)
        # x = F.relu(x1) + F.relu(x3)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x= self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        c = self.lin3(x)
        # c =  F.log_softmax(self.lin3(x), dim=-1)

        return c,x

def convert_pytorch_geometric_databatch_to_dgl_graph(data_batch):

    dgl_graphs = []
    for i in range(len(data_batch)):
        single = data_batch[i]
        # Extract the range of nodes for the current graph
        graph =dgl.graph((single.edge_index[0],single.edge_index[1]),num_nodes= single.num_nodes)
        graph.ndata['feat'] = single.x
        dgl_graphs.append(graph)
    return dgl.batch(dgl_graphs)

# Define a Deep Graph Convolutional Network (DCGCN) model for graph classification
class DCCNN(nn.Module):
    def __init__(self, num_features, num_classes,device):
        super(DCCNN, self).__init__()
        self.device=device
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.jk = JumpingKnowledge(mode='cat')
        self.sortpooling =  SortPooling(k=10)
        self.conv1d1 = torch.nn.Conv1d(10, 32, kernel_size=4, stride=4)
        self.conv1d2 = torch.nn.Conv1d(32, 64, kernel_size=3, stride=3)

        # 1D max pooling
        self.max_pool = torch.nn.MaxPool1d(kernel_size=4)

        # Fully connected layer
        self.fc = nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x1= F.tanh(x1)
        # x = F.relu(x)
        # x = torch.nn.functional.dropout(x, p=0.2, training=self.training)

        x2 = self.conv2(x1, edge_index)
        x2 = F.tanh(x2)
        # x = F.relu(x)
        # x = torch.nn.functional.dropout(x, p=0.2, training=self.training)

        x3 = self.conv3(x2, edge_index)
        x3 = F.tanh(x3)
        # x = F.relu(x)

        x = self.jk([x1, x2, x3])

        g = convert_pytorch_geometric_databatch_to_dgl_graph(data.cpu())
        x = self.sortpooling(g, x.cpu()).to(self.device)
        x = x.view(x.shape[0], 10, 192)
        x = self.conv1d1(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv1d2(x)
        x = F.relu(x)
        x = self.max_pool(x)

        # Flatten output
        x = x.flatten(1)

        # x = torch_geometric.nn.global_mean_pool(x, batch)  # Global pooling for graph-level representation
        c = F.relu(x)
        c = self.fc(c)
        return c,x

class Symmetric(torch.nn.Module):
    def forward(self, w):
        # This class implements the method to define the symmetry in the squared matrices.
        return w.triu(0) + w.triu(1).transpose(-1, -2)
    
class PairwiseParametrization(torch.nn.Module):
    def forward(self, W):
        # Construct a symmetric matrix with zero diagonal
        # The weights are initialized to be non-squared, with 2 additional columns. We cut from two of these
        # two vectors q and r, and then we compute w_diag as described in the paper.
        # This procedure is done in order to easily distribute the mass in its spectrum through the values of q and r
        W0 = W[:, :-2].triu(1)

        W0 = W0 + W0.T

        # Retrieve the `q` and `r` vectors from the last two columns
        q = W[:, -2]
        r = W[:, -1]
        # Construct the main diagonal
        w_diag = torch.diag(q * torch.sum(torch.abs(W0), 1) + r)

        return W0 + w_diag

class External_W(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty((1, input_dim)))
        self.reset_parameters()
        
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.w)

    def forward(self, x):
        # x * self.w behave like a diagonal matrix op., we multiply each row of x by the element-wise w
        return x * self.w


class Source_b(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.empty(1))
     
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.beta)
    


    def forward(self, x):
        return x * self.beta


class PairwiseInteraction_w(nn.Module):
    def __init__(self, input_dim, symmetry_type='1'):
        super().__init__()
        self.W = torch.nn.Linear(input_dim + 2, input_dim, bias = False)

        if symmetry_type == '1':
            symmetry = PairwiseParametrization()
        elif symmetry_type == '2':
            symmetry = Symmetric()

        parametrize.register_parametrization(
            self.W, 'weight', symmetry, unsafe=True)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x):
        return self.W(x)


class GRAFFConv(MessagePassing):
    def __init__(self, external_w, source_b, pairwise_w, self_loops=True):
        super().__init__(aggr='add')

        self.self_loops = self_loops
        self.external_w = external_w #External_W(self.in_dim, device=device)
        self.beta = source_b #Source_b(device = device)
        self.pairwise_W = pairwise_w #PairwiseInteraction_w(self.in_dim, symmetry_type=symmetry_type, device=device)
   

    def forward(self, x, edge_index, x0):

        # We set the source term, which corrensponds with the initial conditions of our system.

        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])

        out_p = self.pairwise_W(x)

        out = self.propagate(edge_index, x=out_p)

        out = out - self.external_w(x) - self.beta(x0)

        return out

    def message(self, x_j, edge_index, x):
        # Does we need the degree of the row or from the columns?
        # x_i are the columns indices, whereas x_j are the row indices
        row, col = edge_index

        # Degree is specified by the row (outgoing edges)
        deg_matrix = degree(col, num_nodes=x.shape[0], dtype=x.dtype)
        deg_inv = deg_matrix.pow(-0.5)
        
        deg_inv[deg_inv == float('inf')] = 0

        denom_degree = deg_inv[row]*deg_inv[col]

        # Each row of denom_degree multiplies (element-wise) the rows of x_j
        return denom_degree.unsqueeze(-1) * x_j

class GINLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,aggr= 'sum'):
        super(GINLayer, self).__init__()
        self.gin_conv = GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        ),aggr = aggr)

    def forward(self, x, edge_index):
        return self.gin_conv(x, edge_index)


class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, dim_in, dim_h, dim_out,graph_pooling='mean', heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h*heads, dim_h, heads=2)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        self.fc1 = nn.Linear(2 * dim_h, dim_h)
        self.fc2 = nn.Linear(dim_h, dim_h // 2)
        self.fc = nn.Linear(dim_h // 2, dim_out)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = self.gat2(h, edge_index)
        h_graph = self.pool(h, batch)
        x = F.relu(self.fc1(h_graph))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        c = self.fc(x)
        return c, x


# Define the GIN model
class GINClassifier_changed(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GINClassifier_changed, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([GINLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)])

        #classifier
        self.fc1 = nn.Linear(hidden_dim*num_layers, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc = nn.Linear(hidden_dim // 2, output_dim)
        self.adder = nn.Linear(hidden_dim, hidden_dim // 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.embedding(x))
        h=[]
        s=[]
        for layer in self.layers:
            x = layer(x, edge_index)
            p = global_add_pool(x, batch)
            s.append(p)
            h.append(p)
        h = torch.cat((h), dim=1)
        # Classifier
        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=0.2, training=self.training)
        h = h+s[1]
        h = self.fc2(h)
        s_2 = self.adder(s[2])
        h = h+s_2
        h = F.relu(h)
        c = self.fc(h)
        return c,h
class GINClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,aggr='sum'):
        super(GINClassifier, self).__init__()
        # self.embedding = torch.nn.Embedding(1, hidden_dim)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([GINLayer(hidden_dim, hidden_dim, hidden_dim,aggr) for _ in range(num_layers)])

        #classifier
        self.fc1 = nn.Linear(hidden_dim*num_layers, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, data,negative = False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.embedding(x))
        h=[]
        for layer in self.layers:
            x = layer(x, edge_index)
            p = global_add_pool(x, batch)
            h.append(p)
        h = torch.cat((h), dim=1)
        # Classifier
        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.fc2(h)
        h = F.relu(h)
        c = self.fc(h)
        return c,h,x

class GCNGraphClassifier(nn.Module):
    def __init__(self, num_node_features, hidden_size, output_size, dropout_rate ,graph_pooling ="mean"):
        super(GCNGraphClassifier, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, 2*hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        self.fc1 = nn.Linear(2*hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc = nn.Linear(hidden_size//2, output_size)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)

        h_graph = self.pool(x, batch)
        x = F.relu(self.fc1(h_graph))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        c = self.fc(x)

            # Global mean pooling over graph nodes
        # x = torch.zeros(data.num_graphs, x.size(1), dtype=x.dtype, device=x.device)
        # for graph_idx in range(data.num_graphs):
        #     mask = (batch == graph_idx)
        #     x[graph_idx] = torch.mean(x[mask], dim=0)

        return c,x

class GRAFFGraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, step = 0.5):
        super(GRAFFGraphClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

        self.external_w = External_W(hidden_dim)
        self.source_b = Source_b()
        self.pairwise_w = PairwiseInteraction_w(
            hidden_dim)

        self.graff = GRAFFConv(self.external_w, self.source_b, self.pairwise_w)
        self.step = step
        self.layers = nn.ModuleList([self.graff for _ in range(num_layers)]) # Weight sharing
        #classifier
        self.fc1 = nn.Linear(hidden_dim*num_layers, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.embedding(x))
        h=[]
        x0 = x.clone()
        for layer in self.layers:
            x = x + self.step*layer(x, edge_index, x0)
            p = global_add_pool(x, batch)
            h.append(p)
        h = torch.cat((h), dim=1)
        # Classifier
        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.fc2(h))
        c = self.fc(h)
        return c,h    


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 32 * 3 * 3)
        x = torch.relu(self.fc1(x))
        y = self.fc2(x)
        return y, x


class CifarCnn(nn.Module):
    def __init__(self, num_classes):
        super(CifarCnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        y = self.fc2(x)
        return y,x
    
class StrongCifarCnn(nn.Module):
    def __init__(self, num_classes):
        super(StrongCifarCnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        y = self.fc3(x)
        return y,x