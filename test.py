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
#MVArandomiser = Randomiser(7,'trk_MVA')

# Create datasets and loaders
test_data = TrackDataset("dataset/Test/test.pkl")
test_dataloader = DataLoader(test_data, batch_size=5000, shuffle=False,num_workers=8)

models_dict = {"pytorch_model":{'model':simpleNN(),'predicted_array':[],'file_location':"model/SavedModels/simplemodel","name":"simple NN"},
               "pytorch_cl_model":{'model':simpleNN(),'predicted_array':[],'file_location':"model/SavedModels/simplemodel_CL","name":"simpleNN replay"},
               "pytorch_cl_model_si":{'model':simpleNN(),'predicted_array':[],'file_location':"model/SavedModels/simplemodel_CL_SI","name":"simpleNN SI"}}


for model in models_dict:

  clf = models_dict[model]["model"]
  clf.load_state_dict(torch.load(models_dict[model]['file_location']))

  true_array = []
  predicted_array = []
  # no need to calculate gradients during inference
  with torch.no_grad():
    for data in test_dataloader:
      inputs, labels = data
      # calculate output by running through the network
      outputs = clf(inputs)
      predicted_array.append(outputs.numpy())
      true_array.append(labels.numpy())

  # Flatten output arrays for ROC plots
  predicted_array = np.concatenate(predicted_array).ravel()
  true_array = np.concatenate(true_array).ravel()

  models_dict[model]['predicted_array'] = predicted_array
  plt.clf()
  figure = plot_split_histo(true_array, predicted_array, model + "Predictions", (0,1),100)
  plt.savefig("%s/%s_PredictionsHisto.png" % ("eval/plots",model))
  plt.close()

plt.clf()
figure = plotPV_roc(true_array,[models_dict[model]["predicted_array"] for model in models_dict],
                               [models_dict[model]["name"] for model in models_dict])
plt.savefig("%s/PVROC.png" % "eval/plots")
plt.close()

