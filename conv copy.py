import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree
from torch_geometric.utils import add_self_loops

import math


class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        


    def forward(self, x, edge_index, edge_attr):
        weight = self.linear.weight.data
        weight = (weight+ weight.T)/2
        self.linear.weight.data = weight
        x = self.linear(x)

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) 
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x,  norm=norm)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    def update(self, aggr_out):
        return aggr_out

#remove once the experiment fails or passes  and uncomment the following function
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)

        

        weight1 = self.mlp[3].weight.data
        eigenvalues1, eigenvectors1 = torch.linalg.eig(weight1)
        real_eigenvalues = eigenvalues1.real
        real_eigenvalues = torch.where(real_eigenvalues < 0, real_eigenvalues, torch.tensor(0.0, device=real_eigenvalues.device))
        diag_real = torch.diag(real_eigenvalues)
        weight1_real = eigenvectors1.real @ diag_real @ torch.linalg.pinv(eigenvectors1.real)
        self.mlp[3].weight.data= weight1_real




        # weight1 = self.mlp[3].weight.data
        # eigenvalues1, eigenvectors1 = torch.linalg.eig(weight1)
        # real_eigenvalues = eigenvalues1.real
        # real_eigenvalues = torch.where(real_eigenvalues > 0, real_eigenvalues, torch.tensor(0.0, device=real_eigenvalues.device))
        # diag_real = torch.diag(real_eigenvalues)
        # weight1_real = eigenvectors1.real @ diag_real @ torch.linalg.pinv(eigenvectors1.real)
        # self.mlp[3].weight.data= weight1_real

        # weight1 = self.mlp[0].weight.data
        # eigenvalues1, eigenvectors1 = torch.linalg.eig(weight1)
        # real_eigenvalues = eigenvalues1.real
        # real_eigenvalues = torch.where(real_eigenvalues > 0, real_eigenvalues, torch.tensor(0.0, device=real_eigenvalues.device))
        # diag_real = torch.diag(real_eigenvalues)
        # weight1_real = eigenvectors1.real @ diag_real @ torch.linalg.pinv(eigenvectors1.real)
        # self.mlp[0].weight.data= weight1_real
        


        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch


        ### computing input node embedding

        h_list = [self.node_encoder(x)]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation



### GIN convolution along the graph structure
#this is just commented for checking the eigens in its square form
# class GINConv(MessagePassing):
#     def __init__(self, emb_dim):
#         '''
#             emb_dim (int): node embedding dimensionality
#         '''

#         super(GINConv, self).__init__(aggr = "add")

#         self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
#         self.eps = torch.nn.Parameter(torch.Tensor([0]))

#         self.edge_encoder = torch.nn.Linear(7, emb_dim)

#     def forward(self, x, edge_index, edge_attr):
#         edge_embedding = self.edge_encoder(edge_attr)
#         out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

#         return out

#     def message(self, x_j, edge_attr):
#         return F.relu(x_j + edge_attr)

#     def update(self, aggr_out):
#         return aggr_out

### GCN convolution along the graph structure
# class GCNConv(MessagePassing):
#     def __init__(self, emb_dim):
#         super(GCNConv, self).__init__(aggr='add')

#         self.linear = torch.nn.Linear(emb_dim, emb_dim)
#         self.root_emb = torch.nn.Embedding(1, emb_dim)
#         self.edge_encoder = torch.nn.Linear(7, emb_dim)

#     def forward(self, x, edge_index, edge_attr):
#         x = self.linear(x)
#         edge_embedding = self.edge_encoder(edge_attr)

#         row, col = edge_index

#         #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
#         deg = degree(row, x.size(0), dtype = x.dtype) + 1
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

#     def message(self, x_j, edge_attr, norm):
#         return norm.view(-1, 1) * F.relu(x_j + edge_attr)

#     def update(self, aggr_out):
#         return aggr_out


### GNN to generate node embedding
# class GNN_node(torch.nn.Module):
#     """
#     Output:
#         node representations
#     """
#     def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
#         '''
#             emb_dim (int): node embedding dimensionality
#             num_layer (int): number of GNN message passing layers

#         '''

#         super(GNN_node, self).__init__()
#         self.num_layer = num_layer
#         self.drop_ratio = drop_ratio
#         self.JK = JK
#         ### add residual connection or not
#         self.residual = residual

#         if self.num_layer < 2:
#             raise ValueError("Number of GNN layers must be greater than 1.")

#         self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding

#         ###List of GNNs
#         self.convs = torch.nn.ModuleList()
#         self.batch_norms = torch.nn.ModuleList()

#         for layer in range(num_layer):
#             if gnn_type == 'gin':
#                 self.convs.append(GINConv(emb_dim))
#             elif gnn_type == 'gcn':
#                 self.convs.append(GCNConv(emb_dim))
#             else:
#                 raise ValueError('Undefined GNN type called {}'.format(gnn_type))

#             self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

#     def forward(self, batched_data):
#         x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch


#         ### computing input node embedding

#         h_list = [self.node_encoder(x)]
#         for layer in range(self.num_layer):

#             h = self.convs[layer](h_list[layer], edge_index, edge_attr)
#             h = self.batch_norms[layer](h)

#             if layer == self.num_layer - 1:
#                 #remove relu for the last layer
#                 h = F.dropout(h, self.drop_ratio, training = self.training)
#             else:
#                 h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

#             if self.residual:
#                 h += h_list[layer]

#             h_list.append(h)

#         ### Different implementations of Jk-concat
#         if self.JK == "last":
#             node_representation = h_list[-1]
#         elif self.JK == "sum":
#             node_representation = 0
#             for layer in range(self.num_layer + 1):
#                 node_representation += h_list[layer]

#         return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))


    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.node_encoder(x)]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation
    


class TransformerConv(torch.nn.Module):
    def __init__(self):
        super(TransformerConv, self).__init__()

    def forward(self, x):
        pass

class GINEConv(torch.nn.Module):
    def __init__(self):
        super(GINEConv, self).__init__()

    def forward(self, x):
        pass

class GATConv(torch.nn.Module):
    def __init__(self):
        super(GATConv, self).__init__()

    def forward(self, x):
        pass

class GATv2Conv(torch.nn.Module):
    def __init__(self):
        super(GATv2Conv, self).__init__()

    def forward(self, x):
        pass

class MFConv(torch.nn.Module):
    def __init__(self):
        super(MFConv, self).__init__()

    def forward(self, x):
        pass

class GENConv(torch.nn.Module):
    def __init__(self):
        super(GENConv, self).__init__()

    def forward(self, x):
        pass

class ExpC_star(torch.nn.Module):
    def __init__(self):
        super(ExpC_star, self).__init__()

    def forward(self, x):
        pass

class ExpC(torch.nn.Module):
    def __init__(self):
        super(ExpC, self).__init__()

    def forward(self, x):
        pass

if __name__ == "__main__":
    pass
