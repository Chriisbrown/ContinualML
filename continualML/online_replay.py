from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import expanduser

import argparse
import torch
from torch.nn import BCELoss
from torchvision import transforms
import torch.optim.lr_scheduler
from avalanche.benchmarks import ni_benchmark
from avalanche.training.supervised.strategy_wrappers_online import OnlineNaive
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.benchmarks.scenarios.online_scenario import OnlineCLScenario
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    confusion_matrix_metrics,
    timing_metrics

)
from avalanche.logging import InteractiveLogger,TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin

from torchvision.datasets import MNIST
from model.simpleNN import simpleNN
import torch
from torchvision import transforms
from dataset.torch_dataset import TrackDataset, Randomiser, GaussianSmear

from metrics import ROC_metrics


z0Smear = GaussianSmear(1,1,'trk_z0')
pTSmear = GaussianSmear(1,1,'trk_pt')
etaSmear = GaussianSmear(1,1,'trk_eta')

transfom_set = transforms.Compose([z0Smear,pTSmear,etaSmear])
# Create datasets and loaders
Traindata_unmodified = TrackDataset("../dataset/Train/train.pkl")
Traindata_smear = TrackDataset("../dataset/Train/train.pkl",transform=transfom_set)

Evaldata_unmodified = TrackDataset("../dataset/Val/val.pkl")
Evaldata_smear = TrackDataset("../dataset/Val/val.pkl",transform=transfom_set)

scenario = ni_benchmark(
        torch.utils.data.ConcatDataset([Traindata_unmodified,Traindata_smear]), 
        torch.utils.data.ConcatDataset([Evaldata_unmodified,Evaldata_smear]), 
        2, task_labels=True, shuffle=False
    )
    # ---------

# MODEL CREATION
model = simpleNN()

# log to Tensorboard
tb_logger = TensorboardLogger()

# log to text file
#text_logger = TextLogger(open('log.txt', 'a'))

# print to stdout
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=False),
    #ROC_metrics(minibatch=True,epoch=True,experience=True,stream=False),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=False),
    timing_metrics(epoch=True, epoch_running=True),
    confusion_matrix_metrics(num_classes=2, save_image=True,
                             stream=True),
    forgetting_metrics(experience=True),
    loggers=[interactive_logger, tb_logger],
    benchmark = scenario
)

    # CREATE THE STRATEGY INSTANCE (ONLINE-REPLAY)
storage_policy = ReservoirSamplingBuffer(max_size=100)
replay_plugin = ReplayPlugin(
        mem_size=100, batch_size=1, storage_policy=storage_policy
    )

cl_strategy = OnlineNaive(
        model,
        torch.optim.Adam(model.parameters(), lr=0.1),
        BCELoss(),
        train_passes=1,
        train_mb_size=1,
        eval_mb_size=32,
        evaluator=eval_plugin,
        plugins=[replay_plugin],
    )

    # TRAINING LOOP
print("Starting experiment...")
results = []

    # Create online benchmark
batch_streams = scenario.streams.values()
    # ocl_benchmark = OnlineCLScenario(batch_streams)
for i, exp in enumerate(scenario.train_stream):
    print("Start of experience: ", exp.current_experience)
    print("Current Classes: ", exp.classes_in_this_experience)
        # Create online scenario from experience exp
    ocl_benchmark = OnlineCLScenario(
            original_streams=batch_streams, experiences=exp, experience_size=1
    )
        # Train on the online train stream of the scenario
    cl_strategy.train(ocl_benchmark.online_train_stream)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.eval(scenario.test_stream))

torch.save(model.state_dict(), "../model/SavedModels/simplemodel_CL_OnlineReplay")