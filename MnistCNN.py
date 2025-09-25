import random
import comet_ml
import argparse
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from datasethelpers.customdataset import CnnMnist, get_random_ids
from loss import ncodLoss
from models.models_layer import SimpleCNN
from utils import set_seed
set_seed()
torch.set_num_threads(1) 

import warnings

warnings.filterwarnings("ignore")
# warnings.filterwarnings("default")
parser = argparse.ArgumentParser(description='GNN baselines on different tasks')
# added by me and can be taken directly to the new settigns
parser.add_argument('--gnn',              type=str,   default='CNN', help='the type of the network needed , (default is GCN)')
parser.add_argument('--device',           type=int,   default=3,         help='which gpu to use if any (default: 0)')
parser.add_argument('--loss',             type=int,   default=1,         help='the loss function that can be used (default: 1, Cross entropy)')
parser.add_argument('--percent',          type=float, default=0.0,       help='percentage of noisy data, (default is no noise)')
parser.add_argument('--lr',               type=float, default=0.001,     help='the learning rate of the network')
parser.add_argument('--lr_u',             type=float, default=1,         help='the learning of the extra parameter')
parser.add_argument('--dataset',          type=str,   default="MNIST",   help='the learning of the extra parameter')
parser.add_argument('--lastlayerdim',     type=int,   default= 64,        help='the dimesions of encoder layer, for ogbg dataset change it to 300,for others 64')
parser.add_argument('--ratio_consitency', type=float, default=0,         help='the ratio at which we want to see the consistency,(default not used )')
parser.add_argument('--ratio_balance',    type=float, default=0,         help='how much balance we want,(default not used again)')
parser.add_argument('--batch_size',       type=int,   default=64,       help='input batch size for training (default: 32),for GNN 128')
parser.add_argument('--epochs',           type=int,   default=1000,      help='number of epochs to train (default: 1000)')
parser.add_argument('--intialbost',       type=int,   default=0,         help='number of epochs you want to run for CE before jumping to our loss function')


args = parser.parse_args()
experiment = Experiment(api_key="your_api_key", project_name="your_project_name", workspace="your_workspace")
device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")


# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = CnnMnist(root='data/Mnist', train=True, download=True, transform=transform)
t_dataset = CnnMnist(root='data/Mnist', train=False, download=True, transform=transform)
# test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_size = int(0.70 * len(dataset))
leave_size = len(dataset) - train_size
test_size = len(t_dataset) 
train_dataset,test_dataset = torch.utils.data.random_split(dataset, [train_size, leave_size])
val_dataset = torch.utils.data.Subset(t_dataset,[i for i in range(0,test_size)])
# val_dataset =test_dataset

args.num_classes = len(dataset.classes)
# args.num_features = dataset.num_features

train_dataset.dataset.train_indices = torch.tensor(train_dataset.indices)
for index, value in enumerate(train_dataset.indices):
    train_dataset.dataset.dict_index[value] = index
train_dataset.dataset.trueLables = train_dataset.dataset.targets[train_dataset.indices].clone()
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
    train_dataset.dataset.targets[p]=i
    allchangedindices = list(set(allchangedindices) - set(p))



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)


train_loss = ncodLoss(train_dataset.dataset.targets[train_dataset.indices],device, num_examp=len(train_dataset.indices),
                      num_classes=args.num_classes,
                      ratio_consistency=args.ratio_consitency, ratio_balance=args.ratio_balance,encoder_features=args.lastlayerdim,total_epochs=args.epochs)
if train_loss.USE_CUDA:
    train_loss.to(device)





pureIndices = []
noisyIndices = []

for x,z in zip(classbins, train_loss.shuffledbins):
    noisyIndices.append(list(set(z) - set(x)))
    pureIndices.append(list(set(z) - (set(z) - set(x))))




# Initialize the model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer_overparametrization = optim.SGD([train_loss.u], lr=args.lr_u)
train_acc_cater = 0.0

experiment.log_parameters(args)
# Training the model
def train(epoch):
    model.train()
    global train_acc_cater,ave_grads, max_grads,train_acc_class
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, data in  enumerate(tqdm(train_loader,  unit='batch')):
        inputs, labels = data[0][0].to(device), data[0][1].to(device)
        target = torch.zeros(len(labels), args.num_classes).to(device).scatter_(1, labels.view(-1, 1).long(), 1)    
        index_run = [train_dataset.dataset.dict_index[int(key)] for key in data[1][0]]

        optimizer.zero_grad()
        optimizer_overparametrization.zero_grad()
        outputs, emb = model(inputs)
        loss = train_loss(index_run, outputs, target, emb, i, epoch,train_acc_cater)
        if ((epoch < args.intialbost) or (args.loss == 1)):
            # loss = F.nll_loss(outputs, labels)
            loss= criterion(outputs, target)
        loss.backward()
        optimizer.step()
        optimizer_overparametrization.step()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels.squeeze()).sum().item()
        running_loss += loss.item()

    experiment.log_metric("Train", (correct_train / total_train), step=epoch)

    train_acc_cater = correct_train / total_train
    
    #remove later this part of the code
    for classNumber, l in zip(range(4), pureIndices[:4]):
        pureAcc = torch.sum(train_loss.take[l]) / len(l)
        experiment.log_metric("pureAcc" + str(classNumber), pureAcc)
    if args.percent > 0:
        for classNumber, l in zip(range(4), noisyIndices[:4]):
            noisyAcc = torch.sum(train_loss.take[l]) / len(l)
            experiment.log_metric("noisyAcc" + str(classNumber), noisyAcc)

    experiment.log_metric("Training Loss", running_loss / len(train_loader))
    return (correct_train / total_train) * 100, (running_loss / len(train_loader))


# Evaluation function
def evaluate(loader, matric_exp):
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, unit='batch')):
            images, labels = data[0][0].to(device), data[0][1].to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels.squeeze()).sum().item()
    experiment.log_metric(matric_exp, (correct_val / total_val), step=epoch)
    return ((correct_val / total_val) * 100)

best_val_accuracy = 0.0
best_model_state_dict = None

for epoch in range(args.epochs):

    train_acc, training_loss = train(epoch)
    val_accuracy = evaluate(val_loader, 'Validation')
    test_accuracy = evaluate(test_loader, 'Test')
    print(
        f"Epoch {epoch + 1}, Test: {test_accuracy :.2f}% , Train: {train_acc :.2f}% ,Val: {val_accuracy :.2f}% ,Loss: {training_loss :.3f}")
   