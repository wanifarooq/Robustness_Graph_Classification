import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size,embedder,device):
        super(SimpleClassifier, self).__init__()
        self.device = device
        self.embedder = embedder
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


    def train_classfier(self,dataloader,val_dataloader, num_epochs=10000, learning_rate=0.001):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.train()
        for epoch in range(num_epochs):
            for data,index in dataloader:
                labels = torch.zeros([data.y.size(0),1], dtype=torch.float,device=self.device)
                _, embedded_data = self.embedder(data.to(self.device))
                optimizer.zero_grad()
                outputs = self(embedded_data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            if epoch % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            accuracy= self.test(val_dataloader)
            print(f'Accuracy: {accuracy:.4f}')
            if loss == 0:
                return
            # if accuracy > 0.9:
            #     print('Best model')
            #     return



    def test(self, dataloader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data,index in dataloader:
                labels = torch.zeros([data.y.size(0),1], dtype=torch.float,device=self.device)
                _ , embedded_data = self.embedder(data.to(self.device))
                outputs = self(embedded_data)
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            accuracy = correct / total
            return accuracy

    def predict(self,dataloader):
        self.eval()
        predictions = []
        index_list=[]
        with torch.no_grad():
            for data,index in dataloader:
                _ , embedded_data = self.embedder(data.to(self.device))
                outputs = self(embedded_data)
                index_run = [dataloader.dataset.dataset.dict_index[int(key)] for key in index[0]]
                # prediction = (outputs > 0.5).float()
                predictions.extend(outputs.view(-1).tolist())
                index_list.extend(index_run)
            return predictions,index_list
