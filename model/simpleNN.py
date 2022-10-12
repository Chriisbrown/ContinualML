import torch
import torch.nn as nn
from dataset.torch_dataset import TrackDataset
# number of features (len of X cols)
input_dim = len(TrackDataset.training_features)
# number of hidden layers
hidden_layers = [25,30]
# number of classes (unique of y)
output_dim = 1

class simpleNN(nn.Module):
  def __init__(self):
    super(simpleNN, self).__init__()
    self.linear1 = nn.Linear(input_dim, hidden_layers[0])
    nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
    self.act1 = nn.ReLU()
    self.linear2 = nn.Linear(hidden_layers[0], hidden_layers[1])
    nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity='relu')
    self.act2 = nn.ReLU()
    self.linear3 = nn.Linear(hidden_layers[1], output_dim)
    nn.init.xavier_uniform_(self.linear3.weight)
    self.act3 = nn.Sigmoid()
    self.double()
  def forward(self, x):
    #x = x.float()
    x = self.linear1(x)
    x = self.act1(x)
    x = self.linear2(x)
    x = self.act2(x)
    x = self.linear3(x)
    x = self.act3(x)
    return x