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

# Create datasets and loaders

list_of_files = ["TTfull","TTv1","TTv2","TTv3","TTv4","TTv5"]

models_dict = {"pytorch_model":{'model':simpleNN(),'predicted_array':[],'rates':[],'file_location':"model/SavedModels/modelTTfull","name":"simple NN unmodifed only"},
              "pytorch_model_retrained":{'model':simpleNN(),'predicted_array':[],'rates':[],'file_location':"model/SavedModels/retrainedmodelTTfull","name":"simple NN retrained"},
              "pytorch_cl_model_si":{'model':simpleNN(),'predicted_array':[],'rates':[],'file_location':"model/SavedModels/modelTTfull_CL_SI","name":"simple NN SI"},
              "pytorch_cl_model_replay":{'model':simpleNN(),'predicted_array':[],'rates':[],'file_location':"model/SavedModels/modelTTfull_CL_Replay","name":"simple NN Replay"}
              }

for i,dataset in enumerate(list_of_files):

  test_data = TrackDataset("dataset/"+dataset+"/Test/test.pkl")
  test_dataloader = DataLoader(test_data, batch_size=5000, shuffle=False,num_workers=8)

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

    rates = calculate_rates(true_array,predicted_array)

    models_dict[model]['predicted_array'].append(predicted_array)
    models_dict[model]['rates'].append(rates)

    plt.clf()
    figure = plot_split_histo(true_array, predicted_array, models_dict[model]["name"] + " Predictions", (0,1),100,
                              model+" Performance On: "+dataset)
    plt.savefig("%s/%s_PredictionsHisto_%s.png" % ("eval/performanceplots",model,dataset))
    plt.close()

  plt.clf()
  figure = plotPV_roc([models_dict[model]["predicted_array"][i] for model in models_dict],
                      [models_dict[model]["rates"][i] for model in models_dict],
                      [models_dict[model]["name"] for model in models_dict],
                      title="All Model's Performance On: "+dataset)
  plt.savefig("%s/PVROC_%s.png" % ("eval/performanceplots",dataset))
  plt.close()


plt.clf()
figure = plotPV_roc(models_dict["pytorch_model"]["predicted_array"],
                    models_dict["pytorch_model"]["rates"],
                    list_of_files,title="Simple Model Initial Training Only")
plt.savefig("%s/PVROC_%s.png" % ("eval/performanceplots","SimpleModelALL"))
plt.close()

plt.clf()
figure = plotPV_roc(models_dict["pytorch_model_retrained"]["predicted_array"],
                    models_dict["pytorch_model_retrained"]["rates"],
                    list_of_files,title="Simple Model Retrained")
plt.savefig("%s/PVROC_%s.png" % ("eval/performanceplots","RetrainedModelALL"))
plt.close()

plt.clf()
figure = plotPV_roc(models_dict["pytorch_cl_model_replay"]["predicted_array"],
                    models_dict["pytorch_cl_model_replay"]["rates"],
                    list_of_files,title="Simple Model Retrained Through Replay")
plt.savefig("%s/PVROC_%s.png" % ("eval/performanceplots","ReplayModelALL"))
plt.close()

plt.clf()
figure = plotPV_roc(models_dict["pytorch_cl_model_si"]["predicted_array"],
                    models_dict["pytorch_cl_model_si"]["rates"],
                    list_of_files, title="Simple Model Retrained Through SI")
plt.savefig("%s/PVROC_%s.png" % ("eval/performanceplots","SIModelALL"))
plt.close()


