from dataset.torch_dataset import EventDataset, Randomiser, GaussianSmear
from torch.utils.data import DataLoader
from model.simpleNN import simpleConv
from eval import *
import torch.nn as nn
import torch
from torchvision import transforms
max_z0 = 20.46912512
nbins = 256

torch.multiprocessing.set_sharing_strategy('file_system')

'''
Testing file for simple model, acts as example of testing a pytorch model
'''

test_data = EventDataset("dataset/TTfull_events/Test/")
test_dataloader = DataLoader(test_data, batch_size=5000, shuffle=False,num_workers=8)


models_dict = {"pytorch_model":{'model':simpleConv(),'predicted_array':[],'file_location':"model/SavedModels/modelTTfull","name":"simple NN umodifed only"},
                }
for model in models_dict:

    clf = models_dict[model]["model"]
    clf.load_state_dict(torch.load(models_dict[model]['file_location']))

    true_array = []
    predicted_array = []
    FH_array = []
    # no need to calculate gradients during inference
    with torch.no_grad():
      for data in test_dataloader:
        inputs, labels = data
        # calculate output by running through the network
        outputs = clf(inputs.float())
        predicted_array.append(outputs.numpy())
        true_array.append(labels.numpy())
        FH_array.append(predictFastHisto(inputs))

    # Flatten output arrays for ROC plots
    predicted_array = np.concatenate(predicted_array).ravel()*max_z0
    true_array = np.concatenate(true_array).ravel()*max_z0
    FH_array = np.concatenate(FH_array).ravel()

    predicted_array = predicted_array - (0.5*(2*max_z0)/nbins )
    models_dict[model]['predicted_array'] = predicted_array

models_dict["FastHisto"] = {'model':None,'predicted_array':FH_array,'file_location':"","name":"Baseline FastHisto"}

plt.clf()
figure = plotz0_residual(true_array,[models_dict[model]["predicted_array"] for model in models_dict],
                                      [models_dict[model]["name"] for model in models_dict])
plt.savefig("%s/PVresidual_%s.png" % ("eval/performanceplots","TTfull_events"))
plt.close()

