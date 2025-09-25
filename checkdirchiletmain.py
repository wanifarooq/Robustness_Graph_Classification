import comet_ml
from comet_ml import Experiment
import glob
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from models.MLP_modules import SimpleClassifier
from datasethelpers.customdataset import CustomizeData, CustomizeDataMnist, get_random_ids, select_random_values, Ogbdataset
import copy
import argparse

from utils.dirchilet import dirichlet_energy
from models.gnn import GNN
from loss import ncodLoss
from utils.Visulaize import visualization, makeImagedir, plot_grad_flow, plot_grad_u
from utils.denserCluster import distribution, makedistdir
from models.models_layer import GINClassifier, DCCNN, HGPSLModel, GCNGraphClassifier, GRAFFGraphClassifier, \
    GINClassifier_changed, GAT
from utils import set_seed
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import DataLoader
# from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import random
# from gnn import GNN
import json

from tqdm.auto import tqdm
import argparse
import time
import numpy as np

### importing OGB
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

set_seed()
torch.set_num_threads(1) 

import warnings

warnings.filterwarnings("ignore")
# warnings.filterwarnings("default")
parser = argparse.ArgumentParser(description='GNN baselines on different tasks')
# added by me and can be taken directly to the new settigns
parser.add_argument('--gnn',              type=str,   default='gin_ogbg_dir_W2', help='the type of the network needed , (default is GCN)')
parser.add_argument('--device',           type=int,   default=2,                    help='which gpu to use if any (default: 0)')
parser.add_argument('--loss',             type=int,   default=1,                    help='the loss function that can be used (default: 1, Cross entropy)')
parser.add_argument('--percent',          type=float, default=0.4,                  help='percentage of noisy data, (default is no noise)')
parser.add_argument('--lr',               type=float, default=0.001,                help='the learning rate of the network')
parser.add_argument('--lr_u',             type=float, default=1,                    help='the learning of the extra parameter')
parser.add_argument('--dataset',          type=str,   default="ogbg-ppa",           help='the learning of the extra parameter')
parser.add_argument('--ratio_consitency', type=float, default=0,                    help='the ratio at which we want to see the consistency,(default not used )')
parser.add_argument('--ratio_balance',    type=float, default=0,                    help='how much balance we want,(default not used again)')
parser.add_argument('--weight_decay',     type=float, default=0.001,                help='weight decay')
parser.add_argument('--nhid',             type=int,   default=128,                  help='hidden size')
parser.add_argument('--sample_neighbor',  type=bool,  default=True,                 help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool,  default=True,                 help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool,default=True,                 help='whether perform structure learning')
parser.add_argument('--pooling_ratio',    type=float, default=0.8,                  help='pooling ratio')
parser.add_argument('--dropout_ratio',    type=float, default=0.5,                  help='dropout ratio, for GNN 0.0')
parser.add_argument('--lamb',             type=float, default=1.0,                  help='trade-off parameter')
parser.add_argument('--num_layer',        type=int,   default=5,                    help='number of GNN message passing layers (default: 3),ogb=5')
parser.add_argument('--emb_dim',          type=int,   default=300,                  help='dimensionality of hidden units in GNNs (default: 300)')
parser.add_argument('--batch_size',       type=int,   default=32,                   help='input batch size for training (default: 32),for GNN ogb = 128')
parser.add_argument('--epochs',           type=int,   default=1000,                 help='number of epochs to train (default: 1000)')
parser.add_argument('--num_workers',      type=int,   default=1,                    help='number of workers (default: 2)')
parser.add_argument('--filename',         type=str,   default="",                   help='filename to output result (default: )')
parser.add_argument('--intialbost',       type=int,   default=0,                    help='number of epochs you want to run for CE before jumping to our loss function')
parser.add_argument('--visual',           type=int,   default=50,                   help='number of samples you want to visualize for each class for each variation i,e noisy and pure')
parser.add_argument('--class2vis',        type=int,   default=4,                    help='number of classes you want to visualize for now max is 4')
parser.add_argument('--lastlayerdim',     type=int,   default=300,                   help='the dimesions of encoder layer, for ogbg dataset change it to 300,for others 64')
parser.add_argument('--patience',         type=int,   default=100,                  help='patience for early stopping')
parser.add_argument('--mlpepochs',        type=int,   default=1000,                 help='epochs for mlp,1000')
parser.add_argument('--mlplr',            type=int,   default=0.001,                help='learning rate for mlp')
parser.add_argument('--aggrNode',         type=str,   default="max",                help='the learning of the extra parameter')
args = parser.parse_args()
experiment = Experiment(api_key="tests", project_name="bests")
# experiment = Experiment(api_key="Iyr1jak4yjocuZKFko2wHRPh3", project_name="noisygraphs",workspace="noisygraphs")
device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data
tag = args.dataset+'_'+str(args.percent)+'_'+str(args.loss)
makeImagedir(tag)
model_dir = makedistdir(tag)



# Load the dataset (e.g., ENZYMES dataset)
if 'ogbg' == args.dataset.split('-')[0]:
    dataset = Ogbdataset(name = args.dataset, transform = add_zeros)
    train_dataset = torch.utils.data.Subset(dataset, dataset.get_idx_split()['train'])
    val_dataset = torch.utils.data.Subset(dataset, dataset.get_idx_split()['valid'])
    test_dataset = torch.utils.data.Subset(dataset, dataset.get_idx_split()['test'])
else:
    dataset = CustomizeData(root='data/' + args.dataset, name=args.dataset, attr=True)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,test_size])
    val_dataset =test_dataset
   

args.num_classes = 2
trainbin = []
# c = [0.12,0.05,0.04,0.05,0.043,0.16]
c = [0.3]*args.num_classes

for i in range(args.num_classes):
    a = torch.nonzero(train_dataset.dataset.y[train_dataset.indices].squeeze() == i).squeeze().tolist()
    l = int(len(a)*c[i])
    trainbin.extend(a[:l])

train_dataset.indices = train_dataset.indices[trainbin]
# train_dataset.indices = torch.tensor(train_dataset.indices)[trainbin].tolist()


valbin = []
for i in range(args.num_classes):
    a= torch.nonzero(train_dataset.dataset.y[val_dataset.indices].squeeze() == i).squeeze().tolist()
    l = int(len(a)*c[i])
    valbin.extend(a[:l])
val_dataset.indices = val_dataset.indices[valbin]
# val_dataset.indices = torch.tensor(val_dataset.indices)[valbin].tolist()


testbin = []
for i in range(args.num_classes):
    a = torch.nonzero(train_dataset.dataset.y[test_dataset.indices].squeeze() == i).squeeze().tolist()
    l = int(len(a)*c[i])
    testbin.extend(a[:l])
test_dataset.indices = test_dataset.indices[testbin]
# test_dataset.indices = torch.tensor(test_dataset.indices)[testbin].tolist()


# args.num_classes = dataset.num_classes
args.num_features = dataset.num_features


train_dataset.dataset.train_indices = torch.tensor(train_dataset.indices)


for index, value in enumerate(train_dataset.indices):
    if 'ogbg' == args.dataset.split('-')[0]:
        train_dataset.dataset.dict_index[value.item()] = index
    else:
        train_dataset.dataset.dict_index[value] = index
train_dataset.dataset.trueLables = train_dataset.dataset.y[train_dataset.indices].clone()
if 'ogbg' == args.dataset.split('-')[0]:
    train_dataset.dataset.trueLables = train_dataset.dataset.trueLables.squeeze()
#Created the classbins to store the indices of the samples for each class,
# note here the indices are the actual ones of the size of the training samples as sent by dataloader not of actual dataset


classbins = []
for i in range(args.num_classes):
    classbins.append(torch.nonzero(train_dataset.dataset.trueLables == i).squeeze().tolist())


changed_indices = []
allchangedindices = []
for index, indexes in enumerate(classbins, 0):
    # get the indices of each class of actual dataset
    class_index = train_dataset.dataset.train_indices[indexes]
    # store the indices of each class corresponding to training of actual dataset before noise
    train_dataset.dataset.class_index.append(class_index)
    changeAbleIndexs= get_random_ids(class_index, args.percent)
    changed_indices.append(changeAbleIndexs.tolist())
    allchangedindices.extend(changeAbleIndexs.tolist())
train_dataset.dataset.changed_indices = changed_indices
for i,x in zip(range(args.num_classes),changed_indices):
    random.shuffle(allchangedindices)
    p = allchangedindices[:len(x)]
    train_dataset.dataset.data.y[p]=i
    allchangedindices = list(set(allchangedindices) - set(p))


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)



train_loss = ncodLoss(train_dataset.dataset.y[train_dataset.indices],device, num_examp=len(train_dataset.indices),
                      num_classes=args.num_classes,
                      ratio_consistency=args.ratio_consitency, ratio_balance=args.ratio_balance,encoder_features=args.lastlayerdim,total_epochs=args.epochs)
if train_loss.USE_CUDA:
    train_loss.to(device)



pureIndices = []
noisyIndices = []
otherClassIndices = []
for x,z in zip(classbins, train_loss.shuffledbins):
    # pureIndices.append(list((set(x) - set(y)) | (set(y).intersection(set(z)))))
    noisyIndices.append(list(set(z) - set(x)))
    pureIndices.append(list(set(z) - (set(z) - set(x))))
    otherClassIndices.append(list(set(x) - set(z)))
if 'ogbg' == args.dataset.split('-')[0]:
    if args.gnn == 'gin_ogbg_dir_W2':
        model = GNN(gnn_type = 'gin', num_class = args.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.dropout_ratio, virtual_node = False).to(device)
        # model1 = GNN(gnn_type = 'gin', num_class = args.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.dropout_ratio, virtual_node = False).to(device)
        # added for test purpose
        # model_sp = GNN(gnn_type = 'gin', num_class = args.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.dropout_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_class = args.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.dropout_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn_ogbg_dir_new':
        model = GNN(gnn_type = 'gcn', num_class = args.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.dropout_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_class = args.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.dropout_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')
else:
    if args.gnn == 'dccnn':
        model = DCCNN(args.num_features, args.num_classes,device).to(device)
    elif args.gnn == 'hgpsl':
        model = HGPSLModel(args).to(device)
    elif args.gnn == 'gin_Enzyme_VPA':
        model = GINClassifier(args.num_features, args.nhid, args.num_classes, args.num_layer, args.aggrNode).to(device)
    elif args.gnn == 'gcn_1':
        model = GCNGraphClassifier(args.num_features, args.nhid, args.num_classes, args.dropout_ratio).to(device)
    elif args.gnn == 'GRAFF':
        print("MODEL IS GRAFF!")
        model = GRAFFGraphClassifier(args.num_features, args.nhid, args.num_classes, num_layers=args.num_layer,
                                     step=0.5).to(device)
    elif args.gnn == 'gin_2':
        model = GINClassifier_changed(args.num_features, args.nhid, args.num_classes, args.num_layer).to(device)
    elif args.gnn == 'gat':
        model = GAT(args.num_features, args.nhid, args.num_classes).to(device)
    elif args.gnn == 'gcn_Enzyme_dir_new':
        model = GNN(gnn_type = 'gcn', num_class = args.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.dropout_ratio, virtual_node = False).to(device)

# criterion = nn.NLLLoss()
# criterion_reductionLess = nn.CrossEntropyLoss(reduction='none')
# optimizer_reductionLess = optim.Adam(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

#added for the test purpose
# optimizer_sp = optim.Adam(model_sp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#optimizer in order to learn the parameter u for each sample
optimizer_overparametrization = optim.SGD([train_loss.u], lr=args.lr_u)


# log the parameters to comet
experiment.log_parameters(args)

# visual = args.visual
# min_vis = min([len(t) for t in noisyIndices])
# if (min_vis < visual) and (args.percent > 0):
#     visual = min_vis


train_acc_cater = 0.0
# Training loop


def train(epoch):
    model.train()
    global train_acc_cater
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, data in  enumerate(tqdm(train_loader,  unit='batch')):
        inputs, labels = data[0].to(device), data[0].y.to(device)
        target = torch.zeros(len(labels), args.num_classes).to(device).scatter_(1, labels.view(-1, 1).long(), 1)    
        if 'ogbg' == args.dataset.split('-')[0]:
            index_run = [train_dataset.dataset.dict_index[int(key.item())] for key in data[1]]
        else:
            index_run = [train_dataset.dataset.dict_index[int(key)] for key in data[1][0]]

        optimizer.zero_grad()
        optimizer_overparametrization.zero_grad()
        outputs, emb,node_embd = model(inputs,True)
        graph_sum= dirichlet_energy(node_embd, inputs.edge_index,inputs._store._slice_dict['edge_index'])
        loss = train_loss(index_run, outputs, target, emb, i, epoch,train_acc_cater,graph_sum)
        if ((epoch < args.intialbost) or (args.loss == 1)):
            loss= criterion(outputs, target)
        loss.backward()
        optimizer.step()
        optimizer_overparametrization.step()
        train_loss.dirchilet[index_run] = torch.stack(graph_sum).detach().view(-1,1)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels.squeeze()).sum().item()
        running_loss += loss.item()
        

        #this part the code have been added to just for purpose of checking the changed loss Method:
        
        # optimizer_reductionLess.zero_grad()
        # pred,_,_ = model1(inputs,False)
        # Effective_weightage = train_loss.take[index_run].detach()
        # reduction_loss  = criterion_reductionLess(pred,target)
        # reduction_loss = Effective_weightage*reduction_loss
        # reduction_loss = reduction_loss.mean()
        # reduction_loss.backward()
        # optimizer_reductionLess.step()


    experiment.log_metric("Train", (correct_train / total_train), step=epoch)
    train_acc_cater = correct_train / total_train

   

    # for con in [4]:
    #     real1 = torch.linalg.eig(model.gnn_node.convs[con].mlp[0].weight.detach()).eigenvalues.real
    #     real2 = torch.linalg.eig(model.gnn_node.convs[con].mlp[3].weight.detach()).eigenvalues.real
    #     positive_sum1 = real1[real1 > 0]
    #     negative_sum1 = real1[real1 < 0]
    #     positive_sum2 = real2[real2 > 0]
    #     negative_sum2 = real2[real2 < 0]
    #     if len(negative_sum1)>0:
    #         experiment.log_metric("w1_layer_negative_sum_" + str(con), abs(negative_sum1.sum()).item())
    #         experiment.log_metric("w1_layer_negative_count_" + str(con),len(negative_sum1))
    #         experiment.log_metric("w1_layer_negative_avg_" + str(con), (abs(negative_sum1.sum()).item())/len(negative_sum1))
    #     if len(negative_sum2)>0:
    #         experiment.log_metric("w2_layer_negative_sum_" + str(con), abs(negative_sum2.sum()).item())
    #         experiment.log_metric("w2_layer_negative_count_" + str(con), len(negative_sum2))
    #         experiment.log_metric("w2_layer_negative_avg_" + str(con), abs(negative_sum2.sum()).item()/len(negative_sum2))

    #     experiment.log_metric("w1_layer_positive_sum_" + str(con), (positive_sum1.sum()).item())
    #     experiment.log_metric("w1_layer_positive_count_" + str(con), len(positive_sum1))
    #     experiment.log_metric("w1_layer_positive_avg_" + str(con), (positive_sum1.sum()).item()/len(positive_sum1))

    #     experiment.log_metric("w2_layer_positive_sum_" + str(con), (positive_sum2.sum()).item())
    #     experiment.log_metric("w2_layer_positive_count_" + str(con), len(positive_sum2))
    #     experiment.log_metric("w2_layer_positive_avg_" + str(con), (positive_sum2.sum()).item()/len(positive_sum2))

        
        # if len(negative_sum1)>0:
        #     experiment.log_metric("w1_layer_negative_1" , torch.topk( abs(negative_sum1), 1)[0][0].item())
        #     experiment.log_metric("w1_layer_negative_2" , torch.topk( abs(negative_sum1), 2)[0][1].item())
        #     experiment.log_metric("w1_layer_negative_3" , torch.topk( abs(negative_sum1), 3)[0][2].item())
        #     experiment.log_metric("w1_layer_negative_4", torch.topk( abs(negative_sum1), 4)[0][3].item())
        #     experiment.log_metric("w1_layer_negative_5" , torch.topk( abs(negative_sum1), 5)[0][4].item())
        # if len(negative_sum2)>0:
        #     experiment.log_metric("w2_layer_negative_1" , torch.topk( abs(negative_sum2), 1)[0][0].item())
        #     experiment.log_metric("w2_layer_negative_2" , torch.topk( abs(negative_sum2), 2)[0][1].item())
        #     experiment.log_metric("w2_layer_negative_3" , torch.topk( abs(negative_sum2), 3)[0][2].item())
        #     experiment.log_metric("w2_layer_negative_4", torch.topk(  abs(negative_sum2), 4)[0][3].item())
        #     experiment.log_metric("w2_layer_negative_5" , torch.topk( abs(negative_sum2), 5)[0][4].item())

        # experiment.log_metric("w1_layer_positive_1" , torch.topk( positive_sum1, 1)[0][0].item())
        # experiment.log_metric("w1_layer_positive_2" , torch.topk( positive_sum1, 2)[0][1].item())
        # experiment.log_metric("w1_layer_positive_3" , torch.topk( positive_sum1, 3)[0][2].item())
        # experiment.log_metric("w1_layer_positive_4" , torch.topk( positive_sum1, 4)[0][3].item())
        # experiment.log_metric("w1_layer_positive_5" , torch.topk( positive_sum1, 5)[0][4].item())

        # experiment.log_metric("w2_layer_positive_1" , torch.topk( positive_sum2, 1)[0][0].item())
        # experiment.log_metric("w2_layer_positive_2" , torch.topk( positive_sum2, 2)[0][1].item())
        # experiment.log_metric("w2_layer_positive_3" , torch.topk( positive_sum2, 3)[0][2].item())
        # experiment.log_metric("w2_layer_positive_4" , torch.topk( positive_sum2, 4)[0][3].item())
        # experiment.log_metric("w2_layer_positive_5" , torch.topk( positive_sum2, 5)[0][4].item())

   
    

    #remove later this part of the code
    for classNumber, l in zip(range(args.num_classes), pureIndices[:args.num_classes]):
        pureAcc = torch.sum(train_loss.take[l]) / len(l)
        experiment.log_metric("pureAcc" + str(classNumber), pureAcc)
    if args.percent > 0:
        for classNumber, l in zip(range(args.num_classes), noisyIndices[:args.num_classes]):
            noisyAcc = torch.sum(train_loss.take[l]) / len(l)
            experiment.log_metric("noisyAcc" + str(classNumber), noisyAcc)



    #for dirchilet energy of each graph
    for classNumber, l in zip(range(args.num_classes), pureIndices[:args.num_classes]):
        pureDir = torch.sum(train_loss.dirchilet[l]) / len(l)
        experiment.log_metric("pureDir" + str(classNumber), pureDir)
    if args.percent > 0:
        for classNumber, l in zip(range(args.num_classes), noisyIndices[:args.num_classes]):
            noisyDir = torch.sum(train_loss.dirchilet[l]) / len(l)
            experiment.log_metric("noisyDir" + str(classNumber), noisyDir)

        for classNumber, l in zip(range(args.num_classes), otherClassIndices[:args.num_classes]):
            noisyOther = torch.sum(train_loss.dirchilet[l]) / len(l)
            experiment.log_metric("noisyDirOther" + str(classNumber), noisyOther)






        # this one uses Tsne to visualize the embeddings
    #     To_visulaize = [sublist1[:visual] + sublist2[:visual] for sublist1, sublist2 in zip(pureIndices[:4], noisyIndices[:4])]
    #     class_seed = [True if item in pure else False for item, pure in zip(seedlist, pureIndices[:4])]
    #     visualization(train_loss.prevSimilarity[sum(To_visulaize, [])], epoch,
    #                   train_loss.prevSimilarity[seedlist], class_seed,True,args.class2vis)

    # else:
    #     To_visulaize = [sublist[:visual] for sublist in pureIndices[:4]]
    #     class_seed = [True for item in seedlist]
    #     visualization(train_loss.prevSimilarity[sum(To_visulaize, [])], epoch,
    #                   train_loss.prevSimilarity[seedlist], class_seed,False,args.class2vis)


    experiment.log_metric("Training Loss", running_loss / len(train_loader))
    return (correct_train / total_train) * 100, (running_loss / len(train_loader))



# def test_trainOnW(epoch):
#     model1.eval()
#     correct_train = 0
#     total_train = 0

#     with torch.no_grad():
#         for i, data in  enumerate(tqdm(train_loader,  unit='batch')):
#             inputs, labels = data[0].to(device), data[0].y.to(device)
#             target = torch.zeros(len(labels), args.num_classes).to(device).scatter_(1, labels.view(-1, 1).long(), 1)    
#             if 'ogbg' == args.dataset.split('-')[0]:
#                 index_run = [train_dataset.dataset.dict_index[int(key.item())] for key in data[1]]
#             else:
#                 index_run = [train_dataset.dataset.dict_index[int(key)] for key in data[1][0]]

#             outputs, emb,node_embd = model1(inputs,True)

#             train_loss(index_run, outputs, target, emb, i, epoch,0,None)
#             _, predicted = torch.max(outputs.data, 1)
#             total_train += labels.size(0)
#             correct_train += (predicted == labels.squeeze()).sum().item()
#     experiment.log_metric("test_train", (correct_train / total_train), step=epoch)
   

#     #remove later this part of the code
#     for classNumber, l in zip(range(4), pureIndices[:4]):
#         pureAcc = torch.sum(train_loss.take[l]) / len(l)
#         experiment.log_metric("pureAcc_testTrain" + str(classNumber), pureAcc)
#     if args.percent > 0:
#         for classNumber, l in zip(range(4), noisyIndices[:4]):
#             noisyAcc = torch.sum(train_loss.take[l]) / len(l)
#             experiment.log_metric("noisyAcc_testTrain" + str(classNumber), noisyAcc)



    









# Evaluation function
def evaluate(loader, matric_exp):
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, unit='batch')):
            images, labels = data[0].to(device), data[0].y.to(device)
            outputs, _ ,_= model(images,False)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels.squeeze()).sum().item()
    experiment.log_metric(matric_exp, (correct_val / total_val), step=epoch)
    return ((correct_val / total_val) * 100)

best_val_accuracy = 0.0
best_model_state_dict = None
patience = args.patience
for epoch in range(args.epochs):

    train_acc, training_loss = train(epoch)


    # model1.load_state_dict(model.state_dict())

    # for layer in range(0,len(model.gnn_node.convs)):
    #     weight1 = model.gnn_node.convs[layer].mlp[3].weight.data
    #     eigenvalues1, eigenvectors1 = torch.linalg.eig(weight1)
    #     real_eigenvalues = eigenvalues1.real
    #     real_eigenvalues = torch.where(real_eigenvalues > 0, real_eigenvalues, torch.tensor(0.0, device=real_eigenvalues.device))
    #     diag_real = torch.diag(real_eigenvalues)
    #     weight1_real = eigenvectors1.real @ diag_real @ torch.linalg.pinv(eigenvectors1.real)
    #     model.gnn_node.convs[layer].mlp[3].weight.data= weight1_real

    # test_trainOnW(epoch)


    val_accuracy = evaluate(val_loader, 'Validation')
    test_accuracy = evaluate(test_loader, 'Test')
    print(
        f"Epoch {epoch + 1}, Test: {test_accuracy :.2f}% , Train: {train_acc :.2f}% ,Val: {val_accuracy :.2f}% ,Loss: {training_loss :.3f}")
    # if val_accuracy > best_val_accuracy:
    #     best_val_accuracy = val_accuracy
    #     best_model_state_dict = model.state_dict()
    #     patience_counter = 0  # Reset patience counter
    # else:
    #     patience_counter += 1

    #     # Check for early stopping
    # if patience_counter >= patience:
    #     print(f"Early stopping at epoch {epoch + 1} due to lack of improvement in validation accuracy.")
    #     break

    # # Save the best model
    # if best_model_state_dict is not None:
    #     torch.save(best_model_state_dict, model_dir+'/embedder.pth')
    #     best_model_state_dict=None
    #     print("Best model saved.",epoch)


# state_path = model_dir+'/embedder.pth'
# state_dict = torch.load(state_path)
# model.load_state_dict(state_dict)
# model.requires_grad_(False)
# mlpclassifier = SimpleClassifier(args.lastlayerdim,args.lastlayerdim,model,device)
# mlpclassifier.to(device)
# # if (args.percent==0.0) and (args.loss==1):
# if (args.percent == 0.0) :
#     train_indices = train_dataset.indices
#     length = len(train_indices)
#     train_dataset_cloned = copy.deepcopy(train_dataset)
#     train_dataset_cloned.indices = train_indices[:(int(length*(2/3)))]
#     val_dataset_cloned = copy.deepcopy(train_dataset)
#     val_dataset_cloned.indices = train_indices[(int(length*(2/3))):]

#     train_loader_cloned = DataLoader(train_dataset_cloned, batch_size=args.batch_size, shuffle=True)
#     val_loader_cloned = DataLoader(val_dataset_cloned, batch_size=args.batch_size, shuffle=False)
#     mlpclassifier.train_classfier(train_loader_cloned,val_loader_cloned,args.mlpepochs,args.mlplr)
#     torch.save(mlpclassifier.state_dict(), model_dir + '/mlp.pth')

# mlp_path = 'model_'+args.dataset+'_0.0_'+args.loss+'/mlp.pth'
# mlp_dict = torch.load(mlp_path)
# mlpclassifier.load_state_dict(mlp_dict)
# pred,indexlist = mlpclassifier.predict(train_loader)
# pred =torch.Tensor(pred)
# classsep={}
# classnumber = 0
# for x,y in zip(pureIndices,noisyIndices):
#     classlist={}
#     classlist['pure']= (pred[[indexlist.index(i) for i in x]].sum() / len(x)).tolist()
#     classlist['noisy'] = (pred[[indexlist.index(i) for i in y]].sum() / len(y)).tolist()
#     classsep[str(classnumber)] = classlist
#     classnumber+=1
# experiment.log_metric("MlpTable", classsep)
# print(classsep)
# file_path = os.path.join(model_dir, f'class_output.json')
# with open(file_path, 'w') as json_file:
#     json.dump(classsep, json_file, indent=2)
