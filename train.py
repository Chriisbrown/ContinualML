from dataset.torch_dataset import TrackDataset, Randomiser
from torch.utils.data import DataLoader
from model.simpleNN import simpleNN
from eval import *
import torch.nn as nn
import torch

'''
Training file for simple model, acts as example of training a pytorch model
'''

# Define any tansformations here
MVArandomiser = Randomiser(7,'trk_MVA')

# Create datasets and creae dataloaders for pytorch
training_data = TrackDataset("dataset/Train/train.pkl",transform=MVArandomiser)
val_data = TrackDataset("dataset/Val/val.pkl",transform=MVArandomiser)

train_dataloader = DataLoader(training_data, batch_size=5000, shuffle=True,num_workers=16)
val_dataloader = DataLoader(val_data, batch_size=5000, shuffle=True,num_workers=16)

# Create model
clf = simpleNN()

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
    # Cast as float (numpy is double by default)
    inputs = inputs.float()
    labels = labels.float()
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

# Save model
torch.save(clf.state_dict(), "model/SavedModels/simplemodel")


