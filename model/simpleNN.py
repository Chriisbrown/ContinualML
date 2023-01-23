import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten


class simpleConv(nn.Module):
  def __init__(self):
    super(simpleConv, self).__init__()

    self.keep_prob = 0.1
     # L1 ImgIn shape=(?, 1, 10, 256)
      # Conv -> (?, 16, , )
      # Pool -> (?, 16, 2, 256)
    self.layer1 = torch.nn.Sequential(
          torch.nn.Conv2d(1, 16, kernel_size=(5,5), stride=(1,1), padding=1),
          #torch.nn.BatchNorm2d(16),
          torch.nn.ReLU(),
          #torch.nn.MaxPool2d(kernel_size=3,stride=1),
          #torch.nn.Dropout(p=1 - self.keep_prob)
          )
      # L2 ImgIn shape=(?, 16, 3, 126)
      # Conv      ->(?, 14, 14, 64)
      # Pool      ->(?, 7, 7, 64)
    self.layer2 = torch.nn.Sequential(
          torch.nn.Conv2d(16, 8, kernel_size=(5,5), stride=1, padding=1),
          #torch.nn.BatchNorm2d(8),
          torch.nn.ReLU(),
          #torch.nn.MaxPool2d(kernel_size=3,stride=1),
          #torch.nn.Dropout(p=1 - self.keep_prob)
          )
      # L3 ImgIn shape=(?, 7, 7, 64)
      # Conv ->(?, 7, 7, 128)
      # Pool ->(?, 4, 4, 128)
    # self.layer3 = torch.nn.Sequential(
    #       torch.nn.Conv2d(8, 4, kernel_size=(3,3), stride=1, padding=1),
    #       #torch.nn.BatchNorm2d(4),
    #       torch.nn.ReLU(),
    #       #torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
    #       torch.nn.Dropout(p=1 - self.keep_prob))

      # L4 FC 4x4x128 inputs -> 625 outputs
    self.fc1 = torch.nn.Linear(12096, 16, bias=True)
    torch.nn.init.xavier_uniform_(self.fc1.weight)
    self.layer4 = torch.nn.Sequential(
          self.fc1,
          #torch.nn.BatchNorm1d(16),
          torch.nn.ReLU())
          #torch.nn.Dropout(p=1 - self.keep_prob))

    self.fc2 = torch.nn.Linear(16, 8, bias=True)
    torch.nn.init.xavier_uniform_(self.fc2.weight)
    self.layer5 = torch.nn.Sequential(
          self.fc2,
          #torch.nn.BatchNorm1d(8),
          torch.nn.ReLU())
          #torch.nn.Dropout(p=1 - self.keep_prob))

    # self.fc3 = torch.nn.Linear(8, 4, bias=True)
    # torch.nn.init.xavier_uniform_(self.fc3.weight)
    # self.layer6 = torch.nn.Sequential(
    #        self.fc3,
    #        torch.nn.BatchNorm1d(4),
    #        torch.nn.ReLU(),
    #        torch.nn.Dropout(p=1 - self.keep_prob))
      # L5 Final FC 625 inputs -> 10 outputs
    self.fc4 = torch.nn.Linear(8, 1, bias=True)
    torch.nn.init.xavier_uniform_(self.fc4.weight) # initialize parameters

  def forward(self, x):
      #print(x.shape)
      out = self.layer1(x)
      #print(out.shape)
      out = self.layer2(out)
      #print(out.shape)
      #out = self.layer3(out)
      out = out.view(out.size(0),-1)
      #print(out.shape)
      out = self.layer4(out)
      #print(out.shape)
      out = self.layer5(out)
      #print(out.shape)
      #out = self.layer6(out)
      out = self.fc4(out)
      #print(out.shape)
      return out