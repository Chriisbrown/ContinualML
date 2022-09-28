from dataset.torch_dataset import TrackDataset
from torch.utils.data import DataLoader
from model.simpleNN import simpleNN
import torch.nn as nn
import torch


training_data = TrackDataset("/home/cebrown/Documents/ContinualAI/dataset/Train/train.pkl")
test_data = TrackDataset("/home/cebrown/Documents/ContinualAI/dataset/Test/test.pkl")
val_data = TrackDataset("/home/cebrown/Documents/ContinualAI/dataset/Val/val.pkl")

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)

clf = simpleNN()

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(clf.parameters(), lr=0.1)

epochs = 2
for epoch in range(epochs):
  running_loss = 0.0
  for i, data in enumerate(train_dataloader, 0):
    inputs, labels = data
    inputs = inputs.float()
    labels = labels.int()
    print(labels)
    # set optimizer to zero grad to remove previous epoch gradients
    optimizer.zero_grad()
    # forward propagation
    outputs = clf(inputs)

    loss = criterion(outputs, labels)
    # backward propagation
    loss.backward()
    # optimize
    optimizer.step()
    running_loss += loss.item()
  # display statistics
  print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')



correct, total = 0, 0
# no need to calculate gradients during inference
with torch.no_grad():
  for data in test_dataloader:
    inputs, labels = data
    # calculate output by running through the network
    outputs = clf(inputs)
    # get the predictions
    __, predicted = torch.max(outputs.data, 1)
    # update results
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print(f'Accuracy of the network on the {len(test_data)} test data: {100 * correct // total} %')
