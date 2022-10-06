from dataset.torch_dataset import TrackDataset, Randomiser
from torch.utils.data import DataLoader
from model.simpleNN import simpleNN
from eval import *
import torch.nn as nn
import torch

'''
Testing file for simple model, acts as example of testing a pytorch model
'''
# Define feature transformations
MVArandomiser = Randomiser(7,'trk_MVA')

# Create datasets and loaders
test_data = TrackDataset("dataset/Test/test.pkl",transform=MVArandomiser)
test_dataloader = DataLoader(test_data, batch_size=5000, shuffle=True,num_workers=16)

clf = simpleNN()
clf.load_state_dict(torch.load("model/SavedModels/simplemodel"))


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

# Flatten output arrays for ROC plots
predicted_array = np.concatenate(predicted_array).ravel()
true_array = np.concatenate(true_array).ravel()

plt.clf()
figure = plotPV_roc(true_array,[predicted_array],["simpleNN"])
plt.savefig("%s/PVROC.png" % "eval/plots")
plt.close()
