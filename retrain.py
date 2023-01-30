from dataset.torch_dataset import EventDataset, Randomiser, GaussianSmear
from torch.utils.data import DataLoader
from model.simpleNN import simpleConv
from eval import *
import torch.nn as nn
from sklearn.metrics import *
import torch
from torchvision import transforms
batchsize = 64
max_z0 = 20.46912512
nbins = 256

'''
Training file for simple model, acts as example of training a pytorch model
'''

list_of_files = ["TTv1full_events","TTv2full_events","TTv3full_events","TTv4full_events","TTv5full_events"]

for i,dataset in enumerate(list_of_files):

  # Create datasets and creae dataloaders for pytorch
  training_data = EventDataset("dataset/"+dataset+"/Train/")
  val_data = EventDataset("dataset/"+dataset+"/Val/")

  train_dataloader = DataLoader(training_data, batch_size=5000, shuffle=True,num_workers=8)
  val_dataloader = DataLoader(val_data, batch_size=5000, shuffle=True,num_workers=8)

  # Create model
  clf = simpleConv()
  clf.load_state_dict(torch.load("model/SavedModels/retrainedmodelTTfull"))

  # Define loss function (Binary Cross Entropy)
  criterion = nn.L1Loss()
  # Define optimisation strategy (Stochastic Gradient Descent) with hyperparameters
  optimizer = torch.optim.SGD(clf.parameters(), lr=0.001,momentum=0.9)

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
      outputs = clf(inputs.float())
      # Calculate loss
      loss = criterion(outputs, labels.float())
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
    torch.save(clf.state_dict(), "model/SavedModels/retrainedmodelTTfull")




