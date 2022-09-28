import torch
import torch.nn as nn

# number of features (len of X cols)
input_dim = 10
# number of hidden layers
hidden_layers = 25
# number of classes (unique of y)
output_dim = 2

class simpleNN(nn.Module):
  def __init__(self):
    super(simpleNN, self).__init__()
    self.linear1 = nn.Linear(input_dim, hidden_layers)
    self.linear2 = nn.Linear(hidden_layers, output_dim)
  def forward(self, x):
    x = self.linear1(x)
    x = torch.relu(x)
    x = self.linear2(x)
    return x