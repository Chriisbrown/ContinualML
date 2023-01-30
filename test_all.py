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

# Create datasets and loaders

list_of_files = ["TTfull_events","TTv1full_events","TTv2full_events","TTv3full_events","TTv4full_events","TTv5full_events"]

models_dict = {"pytorch_model":{'model':simpleConv(),'predicted_array':[],'file_location':"model/SavedModels/modelTTfull","name":"NN Baseline"},
               "pytorch_model_retrained":{'model':simpleConv(),'predicted_array':[],'file_location':"model/SavedModels/retrainedmodelTTfull","name":"NN Retrained"},
               "FastHisto": {'model':None,'predicted_array':[],'file_location':"","name":"Baseline FastHisto"}
               #"pytorch_cl_model_si":{'model':simpleNN(),'predicted_array':[],'rates':[],'file_location':"model/SavedModels/modelTTfull_CL_SI","name":"NN SI"},
               #"pytorch_cl_model_replay":{'model':simpleNN(),'predicted_array':[],'rates':[],'file_location':"model/SavedModels/modelTTfull_CL_Replay","name":"NN Replay"}
              }

for i,dataset in enumerate(list_of_files):

  test_data = EventDataset("dataset/"+dataset+"/Test/")
  test_dataloader = DataLoader(test_data, batch_size=5000, shuffle=False,num_workers=8)

  for model in models_dict:

    if model != "FastHisto":

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

      predicted_array = predicted_array - (0.5*(2*max_z0)/nbins )
      models_dict[model]['predicted_array'].append(predicted_array)

    else:

      FH_array = np.concatenate(FH_array).ravel()
      models_dict["FastHisto"]['predicted_array'].append(FH_array)


  plt.clf()
  figure = plotz0_residual(true_array,[models_dict[model]["predicted_array"] for model in models_dict], 
                                      [models_dict[model]["name"] for model in models_dict],                    
                                      "All Model's Performance On: "+dataset)
  plt.savefig("%s/PVresidual_%s.png" % ("eval/performanceplots",dataset))
  plt.close()

  plt.clf()
  figure = plotz0_residual(true_array,FH_array,  "FastHisto Predictions",
                                        "FastHisto Performance On: "+dataset)
  plt.savefig("%s/%s_PVresidual_%s.png" % ("eval/performanceplots",model,dataset))
  plt.close()


plt.clf()
figure = plotz0_residual(true_array,models_dict["pytorch_model"]["predicted_array"], 
                                     list_of_files,                    
                                      "Baseline Model")
plt.savefig("%s/PVresidual_%s.png" % ("eval/performanceplots","BaselineModelALL"))
plt.close()

plt.clf()
figure = plotz0_residual(true_array,models_dict["pytorch_model_retrained"]["predicted_array"], 
                                     list_of_files,                    
                                      "Model Retrained")
plt.savefig("%s/PVresidual_%s.png" % ("eval/performanceplots","RetrainedModelALL"))
plt.close()

# plt.clf()
# figure = plotPV_roc(models_dict["pytorch_cl_model_replay"]["rates"],
#                     list_of_files,title="Model Retrained Through Replay")
# plt.savefig("%s/PVROC_%s.png" % ("eval/performanceplots","ReplayModelALL"))
# plt.close()

# plt.clf()
# figure = plotPV_roc(models_dict["pytorch_cl_model_si"]["rates"],
#                     list_of_files, title="Model Retrained Through SI")
# plt.savefig("%s/PVROC_%s.png" % ("eval/performanceplots","SIModelALL"))
# plt.close()


