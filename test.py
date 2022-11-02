from dataset.torch_dataset import TrackDataset, Randomiser, GaussianSmear
from torch.utils.data import DataLoader
from model.simpleNN import simpleNN
from eval import *
import torch.nn as nn
import torch
from torchvision import transforms

torch.multiprocessing.set_sharing_strategy('file_system')

'''
Testing file for simple model, acts as example of testing a pytorch model
'''
# Define feature transformations
z0Smear = GaussianSmear(1,1,'trk_z0')
pTSmear = GaussianSmear(1,1,'trk_pt')
etaSmear = GaussianSmear(1,1,'trk_eta')

transfom_set = transforms.Compose([z0Smear,pTSmear,etaSmear])
# Create datasets and loaders
test_data_unmodified = TrackDataset("dataset/Test/test.pkl")
test_dataloader_unmodified = DataLoader(test_data_unmodified, batch_size=5000, shuffle=False,num_workers=8)

test_data_smear = TrackDataset("dataset/Test/test.pkl")
test_dataloader_smear = DataLoader(test_data_smear, batch_size=5000, shuffle=False,num_workers=8)

dataloaders = [test_dataloader_unmodified,test_dataloader_smear]
datanames = ["unmodified","smeared"]

for dataloader,dataname in zip(dataloaders,datanames):
  models_dict = {"pytorch_model":{'model':simpleNN(),'predicted_array':[],'file_location':"model/SavedModels/simplemodel","name":"simple NN unmodifed only"},
               "pytorch_model_smear":{'model':simpleNN(),'predicted_array':[],'file_location':"model/SavedModels/simplemodel_smear","name":"simple NN smear only"},
               "pytorch_model_retrained":{'model':simpleNN(),'predicted_array':[],'file_location':"model/SavedModels/simplemodel_retrain","name":"simple NN retrained on smear"},
               #"pytorch_cl_model":{'model':simpleNN(),'predicted_array':[],'file_location':"model/SavedModels/simplemodel_CL_Replay","name":"simpleNN replay"},
               "pytorch_cl_model_si":{'model':simpleNN(),'predicted_array':[],'file_location':"model/SavedModels/simplemodel_CL_SI","name":"simpleNN SI"}
               }
  for model in models_dict:

    clf = models_dict[model]["model"]
    clf.load_state_dict(torch.load(models_dict[model]['file_location']))

    true_array = []
    predicted_array = []
    # no need to calculate gradients during inference
    with torch.no_grad():
      for data in dataloader:
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
    plt.savefig("%s/%s_PredictionsHisto_%s.png" % ("eval/performanceplots",model,dataname))
    plt.close()

  plt.clf()
  figure = plotPV_roc(true_array,[models_dict[model]["predicted_array"] for model in models_dict],
                                [models_dict[model]["name"] for model in models_dict])
  plt.savefig("%s/PVROC_%s.png" % ("eval/performanceplots",dataname))
  plt.close()

