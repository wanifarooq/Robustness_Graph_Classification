import comet_ml
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
from datasethelpers.customdataset import CustomizeData, Ogbdataset, get_random_ids
from models.gnn import GNN
from loss1 import ncodLoss
from utils.Visulaize import visualization, makeImagedir, plot_grad_flow
from utils.denserCluster import distribution, makedistdir
from models.models_layer import GINClassifier, DCCNN, HGPSLModel, GCNGraphClassifier, GRAFFGraphClassifier, \
    GINClassifier_changed, GAT
from utils import set_seed
from torch_geometric.loader import DataLoader
import argparse
import random


set_seed()
torch.set_num_threads(1)
import warnings

warnings.filterwarnings("ignore")
# warnings.filterwarnings("default")
parser = argparse.ArgumentParser(description='GNN baselines on different tasks')
# added by me and can be taken directly to the new settigns
parser.add_argument('--gnn',              type=str,   default='gin_actual',     help='the type of the network needed , (default is GCN)')
parser.add_argument('--device',           type=int,   default=0,         help='which gpu to use if any (default: 0)')
parser.add_argument('--loss',             type=int,   default=0,         help='the loss function that can be used (default: 1, Cross entropy)')
parser.add_argument('--percent',          type=float, default=0.2,       help='percentage of noisy data, (default is no noise)')
parser.add_argument('--lr',               type=float, default=0.001,     help='the learning rate of the network')
parser.add_argument('--lr_u',             type=float, default=1,         help='the learning of the extra parameter')
parser.add_argument('--dataset',          type=str,   default="ENZYMES", help='the learning of the extra parameter')
parser.add_argument('--ratio_consitency', type=float, default=0,         help='the ratio at which we want to see the consistency,(default not used )')
parser.add_argument('--ratio_balance',    type=float, default=0,         help='how much balance we want,(default not used again)')
parser.add_argument('--weight_decay',     type=float, default=0.001,     help='weight decay')
parser.add_argument('--nhid',             type=int,   default=128,       help='hidden size')
parser.add_argument('--sample_neighbor',  type=bool,  default=True,      help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool,  default=True,      help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool,default=True,      help='whether perform structure learning')
parser.add_argument('--pooling_ratio',    type=float, default=0.8,       help='pooling ratio')
parser.add_argument('--dropout_ratio',    type=float, default=0.5,       help='dropout ratio, for GNN 0.0')
parser.add_argument('--lamb',             type=float, default=1.0,       help='trade-off parameter')
parser.add_argument('--num_layer',        type=int,   default=3,         help='number of GNN message passing layers (default: 3)')
parser.add_argument('--emb_dim',          type=int,   default=300,       help='dimensionality of hidden units in GNNs (default: 300)')
parser.add_argument('--batch_size',       type=int,   default=128,       help='input batch size for training (default: 32),for GNN 128')
parser.add_argument('--epochs',           type=int,   default=1000,      help='number of epochs to train (default: 1000)')
parser.add_argument('--num_workers',      type=int,   default=2,         help='number of workers (default: 2)')
parser.add_argument('--filename',         type=str,   default="",        help='filename to output result (default: )')
parser.add_argument('--intialbost',       type=int,   default=0,         help='number of epochs you want to run for CE before jumping to our loss function')
parser.add_argument('--visual',           type=int,   default=50,        help='number of samples you want to visualize for each class for each variation i,e noisy and pure')
parser.add_argument('--class2vis',        type=int,   default=4,         help='number of classes you want to visualize for now max is 4')
parser.add_argument('--lastlayerdim',     type=int,   default=64,        help='the dimesions of encoder layer, for ogbg dataset change it to 300,for others 64')
parser.add_argument('--patience',         type=int,   default=1000,       help='patience for early stopping')
parser.add_argument('--mlpepochs',        type=int,   default=1000,       help='epochs for mlp,1000')
parser.add_argument('--mlplr',            type=int,   default=0.001,      help='learning rate for mlp')






args = parser.parse_args()
experiment = Experiment(api_key="your_api_key", project_name="your_project_name", workspace="your_workspace")
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

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features
train_dataset.dataset.train_indices = torch.tensor(train_dataset.indices)

for index, value in enumerate(train_dataset.indices):
    train_dataset.dataset.dict_index[value] = index
train_dataset.dataset.trueLables = train_dataset.dataset.y[train_dataset.indices].clone()

classbins = []
for i in range(dataset.num_classes):
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
                      num_classes=dataset.num_classes,
                      ratio_consistency=args.ratio_consitency, ratio_balance=args.ratio_balance,encoder_features=args.lastlayerdim,total_epochs=args.epochs)
if train_loss.USE_CUDA:
    train_loss.to(device)


pureIndices = []
noisyIndices = []

for x, z in zip(classbins, train_loss.shuffledbins):
    # pureIndices.append(list((set(x) - set(y)) | (set(y).intersection(set(z)))))
    noisyIndices.append(list(set(z) - set(x)))
    pureIndices.append(list(set(z) - (set(z) - set(x))))

if 'ogbg' == args.dataset.split('-')[0]:
    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.dropout_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.dropout_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.dropout_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.dropout_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')
else:
    if args.gnn == 'dccnn':
        model = DCCNN(args.num_features, args.num_classes,device).to(device)
    elif args.gnn == 'hgpsl':
        model = HGPSLModel(args).to(device)
    elif args.gnn == 'gin_actual':
        model = GINClassifier(args.num_features, args.nhid, args.num_classes, args.num_layer).to(device)
    elif args.gnn == 'gcn_actual':
        model = GCNGraphClassifier(args.num_features, args.nhid, args.num_classes, args.dropout_ratio).to(device)
    elif args.gnn == 'GRAFF':
        print("MODEL IS GRAFF!")
        model = GRAFFGraphClassifier(args.num_features, args.nhid, args.num_classes, num_layers=args.num_layer,
                                     step=0.5).to(device)
    elif args.gnn == 'gin_actual_changed':
        model = GINClassifier_changed(args.num_features, args.nhid, args.num_classes, args.num_layer).to(device)
    elif args.gnn == 'gat_actual':
        model = GAT(args.num_features, args.nhid, args.num_classes).to(device)

# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

#optimizer in order to learn the parameter u for each sample
optimizer_overparametrization = optim.SGD([train_loss.u], lr=args.lr_u)


# log the parameters to comet
experiment.log_parameters(args)

visual = args.visual
min_vis = min([len(t) for t in noisyIndices])
if (min_vis < visual) and (args.percent > 0):
    visual = min_vis

train_acc_class = torch.zeros((args.num_classes), device=device)
train_acc_cater = 0.0
# Training loop

ave_grads = []
max_grads = []
def calculate_grad_u(named_parameters,index_run):
    ave_grads.append(named_parameters.grad[index_run].abs().mean().cpu())
    max_grads.append(named_parameters.grad[index_run].abs().max().cpu())

def train(epoch):
    model.train()
    global train_acc_cater,ave_grads, max_grads,train_acc_class
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[0].y.to(device)
        target = torch.zeros(len(labels), args.num_classes).to(device).scatter_(1, labels.view(-1, 1).long(), 1)
        index_run = [train_dataset.dataset.dict_index[int(key)] for key in data[1][0]]
        outputs, emb = model(inputs)
        optimizer.zero_grad()
        optimizer_overparametrization.zero_grad()
        loss = train_loss(index_run, outputs, target, emb, i, epoch,train_acc_cater)
        if ((epoch < args.intialbost) or (args.loss == 1)):
            # loss = F.nll_loss(outputs, labels)
            loss= criterion(outputs, labels)
        loss.backward()

#  these are for visualization only

        # calculate_grad_u(train_loss.u,index_run)
        # if i==3:
        #     plot_grad_flow(model.named_parameters(),epoch)

        optimizer.step()
        optimizer_overparametrization.step()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        running_loss += loss.item()
    experiment.log_metric("Train", (correct_train / total_train), step=epoch)
    train_acc_cater = correct_train / total_train

#  this is necessary for the above things

    # plot_grad_u(ave_grads, max_grads,epoch)
    # ave_grads = []
    # max_grads = []

    seedlist = []
    distanceofseeds = []
    distanceFromOthers = []
    for sublist in train_loss.shuffledbins:
        vectors = train_loss.prevSimilarity[sublist]
        distances = torch.cdist(vectors, vectors, p=2)
        hdv = torch.argmin(torch.sum(torch.sort(distances, axis=1)[0][:, :50 ], axis=1))
        seedlist.append(sublist[hdv.item()])
        distanceofseeds.append(distances[hdv])
        NotinsubList = [x for x in train_dataset.dataset.dict_index.values() if x not in sublist]
        # NotinsubList = random.sample(NotinsubList, len(distances))
        vectors = train_loss.prevSimilarity[NotinsubList]
        vector = train_loss.prevSimilarity[sublist[hdv.item()]]
        distanceFromOthers.append(torch.cdist(vector.view(1, args.lastlayerdim), vectors).view(-1))
    try:
        distribution(distanceofseeds,distanceFromOthers, epoch, 'own-other',args.class2vis)
    except Exception as e:
        print(f"An error occurred: {e}")
        print(distanceofseeds, distanceFromOthers)
        print(vectors)



    #  This is used to get the accuracy of puresamples and the noisy samples.
    for classNumber, l in zip(range(train_loss.num_classes), pureIndices):
        pureAcc = torch.sum(train_loss.take[l]) / len(l)
        experiment.log_metric("pureAcc" + str(classNumber), pureAcc)
    if args.percent > 0:
        for classNumber, l in zip(range(train_loss.num_classes), noisyIndices):
            noisyAcc = torch.sum(train_loss.take[l]) / len(l)
            experiment.log_metric("noisyAcc" + str(classNumber), noisyAcc)

        # this one uses Tsne to visualize the embeddings
        To_visulaize = [sublist1[:visual] + sublist2[:visual] for sublist1, sublist2 in zip(pureIndices, noisyIndices)]
        class_seed = [True if item in pure else False for item, pure in zip(seedlist, pureIndices)]
        visualization(train_loss.prevSimilarity[sum(To_visulaize, [])], epoch,
                      train_loss.prevSimilarity[seedlist], class_seed,True,args.class2vis)

    else:
        To_visulaize = [sublist[:visual] for sublist in pureIndices]
        class_seed = [True for item in seedlist]
        visualization(train_loss.prevSimilarity[sum(To_visulaize, [])], epoch,
                      train_loss.prevSimilarity[seedlist], class_seed,False,args.class2vis)

    # These are to get the distribution if there is added noise to the training data
    if args.percent >0:
        pureDistance = []
        noiseDistance = []

        for vector1, vector2, vector3 in zip(distanceofseeds, noisyIndices, train_loss.shuffledbins):
            x = [element in vector2 for element in vector3]
            noiseDistance.append(vector1[x])
            inver_x = [not value for value in x]
            pureDistance.append(vector1[inver_x])
        distribution(pureDistance,noiseDistance, epoch, 'pure-noise',args.class2vis)
        # distribution(noiseDistance, epoch, 'noise',args.class2vis)

    experiment.log_metric("Training Loss", running_loss / len(train_loader))
    return (correct_train / total_train) * 100, (running_loss / len(train_loader))


# Evaluation function
def evaluate(loader, matric_exp):
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[0].y.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    experiment.log_metric(matric_exp, (correct_val / total_val), step=epoch)
    return ((correct_val / total_val) * 100)

best_val_accuracy = 0.0
best_model_state_dict = None
patience = args.patience
for epoch in range(args.epochs):

    train_acc, training_loss = train(epoch)
    val_accuracy = evaluate(val_loader, 'Validation')
    test_accuracy = evaluate(test_loader, 'Test')
    print(
        f"Epoch {epoch + 1}, Test: {test_accuracy :.2f}% , Train: {train_acc :.2f}% ,Val: {val_accuracy :.2f}% ,Loss: {training_loss :.3f}")
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state_dict = model.state_dict()
        patience_counter = 0  # Reset patience counter
    else:
        patience_counter += 1

        # Check for early stopping
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch + 1} due to lack of improvement in validation accuracy.")
        break

    # Save the best model
    if best_model_state_dict is not None:
        torch.save(best_model_state_dict, model_dir+'/embedder.pth')
        best_model_state_dict=None
        print("Best model saved.",epoch)

