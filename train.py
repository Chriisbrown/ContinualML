from dataset.torch_dataset import TrackDataset
from torch.utils.data import DataLoader
from model.simpleNN import simpleNN
from eval import *
import torch.nn as nn
import torch


training_data = TrackDataset("/home/cebrown/Documents/ContinualAI/dataset/Train/train.pkl")
test_data = TrackDataset("/home/cebrown/Documents/ContinualAI/dataset/Test/test.pkl")
val_data = TrackDataset("/home/cebrown/Documents/ContinualAI/dataset/Val/val.pkl")

train_dataloader = DataLoader(training_data, batch_size=5000, shuffle=True,num_workers=16)
test_dataloader = DataLoader(test_data, batch_size=5000, shuffle=True,num_workers=16)
val_dataloader = DataLoader(val_data, batch_size=5000, shuffle=True,num_workers=16)

clf = simpleNN()

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(clf.parameters(), lr=0.01,momentum=0.9)

epochs = 20
for epoch in range(epochs):
  running_loss = 0.0
  for i, data in enumerate(train_dataloader, 0):
    inputs, labels = data
    inputs = inputs.float()
    labels = labels.float()
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



predicted_array = []
true_array = []
# no need to calculate gradients during inference
with torch.no_grad():
  for data in test_dataloader:
    inputs, labels = data
    inputs = inputs.float()
    labels = labels.float()
    # calculate output by running through the network
    outputs = clf(inputs)
    predicted_array.append(outputs.numpy())
    true_array.append(labels.numpy())

predicted_array = np.concatenate(predicted_array).ravel()
true_array = np.concatenate(true_array).ravel()

plt.clf()
figure=plotPV_roc(true_array,predicted_array,"simpleNN")
plt.savefig("%s/PVROC.png" % "plots")
plt.close()



