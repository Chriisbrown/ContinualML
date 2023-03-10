from dataset.torch_dataset import TrackDataset, Randomiser, GaussianSmear
from torch.utils.data import DataLoader
from model.simpleNN import simpleNN
from eval import *
import torch.nn as nn
import torch
from torchvision import transforms

'''
Training file for simple model, acts as example of training a pytorch model
'''

list_of_files = ["TTv1","TTv2","TTv3","TTv4","TTv5"]

for i,dataset in enumerate(list_of_files):

  # Create datasets and creae dataloaders for pytorch
  training_data = TrackDataset("dataset/"+dataset+"/Train/train.pkl")
  val_data = TrackDataset("dataset/"+dataset+"/Val/val.pkl")

  train_dataloader = DataLoader(training_data, batch_size=5000, shuffle=True,num_workers=8)
  val_dataloader = DataLoader(val_data, batch_size=5000, shuffle=True,num_workers=8)

  # Create model
  clf = simpleNN()
  clf.load_state_dict(torch.load("model/SavedModels/retrainedmodelTTfull"))

  # Define loss function (Binary Cross Entropy)
  criterion = nn.BCELoss()
  # Define optimisation strategy (Stochastic Gradient Descent) with hyperparameters
  optimizer = torch.optim.SGD(clf.parameters(), lr=0.01,momentum=0.9)

  epochs = 20
  # Iterate through training epochs (one full round of entire training dataset)
  for epoch in range(epochs):
    running_loss = 0.0
    # Iterate through batches
    for i, data in enumerate(train_dataloader, 0):
      inputs, labels = data
      # set optimizer to zero grad to remove previous epoch gradients
      optimizer.zero_grad()
      # forward propagation
      outputs = clf(inputs)
      # Calculate loss
      loss = criterion(outputs, labels)
      # backward propagation
      loss.backward()
      # optimize
      optimizer.step()
      running_loss += loss.item()
    # display statistics
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
    torch.save(clf.state_dict(), "model/SavedModels/retrainedmodelTTfull")




