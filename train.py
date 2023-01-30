from dataset.torch_dataset import EventDataset, Randomiser, GaussianSmear
from torch.utils.data import DataLoader
from model.simpleNN import simpleConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from eval import *
import torch.nn as nn
from sklearn.metrics import *
import torch
from torchvision import transforms
batchsize = 64
max_z0 = 20.46912512
nbins = 256

torch.multiprocessing.set_sharing_strategy('file_system')
'''
Training file for simple model, acts as example of training a pytorch model
'''
models_dict = {"pytorch_model":{'model':simpleConv(),'predicted_array':[],'file_location':"model/SavedModels/modelTTfull","name":"simple NN umodifed only"},
                }

# Create datasets and creae dataloaders for pytorch
training_data = EventDataset("dataset/TTfull_events/Train/")
val_data = EventDataset("dataset/TTfull_events/Val/")

train_dataloader = DataLoader(training_data, batch_size=batchsize, shuffle=True,num_workers=16,drop_last=True )
val_dataloader = DataLoader(val_data, batch_size=batchsize, shuffle=True,num_workers=16,drop_last=True )

# Create model
clf = simpleConv()
clf = clf.float()

# Define loss function (Binary Cross Entropy)
#criterion = nn.HuberLoss(delta=0.6)
criterion = nn.L1Loss()
# Define optimisation strategy (Stochastic Gradient Descent) with hyperparameters
#optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)

optimizer = torch.optim.SGD(clf.parameters(), lr=0.01, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min',verbose=True,patience=3)

epochs = 75
# Iterate through training epochs (one full round of entire training dataset)
for epoch in range(epochs):
  running_loss = 0.0
  # Iterate through batches
  for i, data in enumerate(train_dataloader, 0):
    inputs, labels = data
    # set optimizer to zero grad to remove previous epoch gradients
    optimizer.zero_grad()
    # forward propagation
    outputs = clf(inputs.float())
    #print(outputs,labels)
    # Calculate loss
    loss = criterion(outputs,labels.float())
    # backward propagation
    loss.backward()
    # optimize
    optimizer.step()
    running_loss += loss.item()
  # display statistics
  print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batchsize:.5f}')

  true_array = []
  predicted_array = []
  FH_array = []
  with torch.no_grad():
      for data in val_dataloader:
        inputs, labels = data
        # calculate output by running through the network
        outputs = clf(inputs.float())
        predicted_array.append(outputs.numpy())
        true_array.append(labels.numpy())
        FH_array.append(predictFastHisto(inputs))

  predicted_array = np.concatenate(predicted_array).ravel()*max_z0
  true_array = np.concatenate(true_array).ravel()*max_z0
  FH_array = np.concatenate(FH_array).ravel()

  predicted_array = predicted_array - (0.5*(2*max_z0)/nbins )

  print(f'[{epoch + 1}, {i + 1:5d}] MSE: {mean_squared_error(true_array, predicted_array):.5f} FH MSE: {mean_squared_error(true_array, FH_array):.5f}')
  print(f'[{epoch + 1}, {i + 1:5d}] MAE: {mean_absolute_error(true_array, predicted_array):.5f} FH MAE: {mean_absolute_error(true_array, FH_array):.5f}')
  print(f'[{epoch + 1}, {i + 1:5d}] R2:  {r2_score(true_array, predicted_array):.5f} FH R2:  {r2_score(true_array, FH_array):.5f}')
  print("=============================================")


  # Save model
  torch.save(clf.state_dict(),  models_dict["pytorch_model"]['file_location'])


