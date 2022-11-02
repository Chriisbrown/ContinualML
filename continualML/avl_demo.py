
from torch.optim import SGD,Adam
from torch.nn import CrossEntropyLoss, BCELoss
from dataset.torch_dataset import TrackDataset, Randomiser, GaussianSmear
from torch.utils.data import DataLoader
from avalanche.benchmarks import ni_benchmark, generators, utils
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.scenarios import OnlineCLScenario
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive

from torchvision.datasets import MNIST
from model.simpleNN import simpleNN
import torch
from torchvision import transforms

from metrics import ROC_metrics
from strategies import ReplayP

from avalanche.training.supervised.strategy_wrappers import SynapticIntelligence

# Create datasets and loaders
Traindata = TrackDataset("../dataset/TTfull/Train/train.pkl")
Traindata_v1 = TrackDataset("../dataset/TTv1/Train/train.pkl")
Traindata_v2 = TrackDataset("../dataset/TTv2/Train/train.pkl")
Traindata_v3 = TrackDataset("../dataset/TTv3/Train/train.pkl")
Traindata_v4 = TrackDataset("../dataset/TTv4/Train/train.pkl")
Traindata_v5 = TrackDataset("../dataset/TTv5/Train/train.pkl")

Evaldata = TrackDataset("../dataset/TTfull/Val/val.pkl")
Evaldata_v1 = TrackDataset("../dataset/TTv1/Val/val.pkl")
Evaldata_v2 = TrackDataset("../dataset/TTv2/Val/val.pkl")
Evaldata_v3 = TrackDataset("../dataset/TTv3/Val/val.pkl")
Evaldata_v4 = TrackDataset("../dataset/TTv4/Val/val.pkl")
Evaldata_v5 = TrackDataset("../dataset/TTv5/Val/val.pkl")


train_fixed_exp_assignment = [[i for i in range(len(Traindata))],
                              [i + len(Traindata) for i in range(len(Traindata_v1))],
                              [i + len(Traindata) + len(Traindata_v1) for i in range(len(Traindata_v2))],
                              [i + len(Traindata) + len(Traindata_v1) + len(Traindata_v2) for i in range(len(Traindata_v3))],
                              [i + len(Traindata) + len(Traindata_v1) + len(Traindata_v2) + len(Traindata_v4) for i in range(len(Traindata_v4))],
                              [i + len(Traindata) + len(Traindata_v1) + len(Traindata_v2) + len(Traindata_v4) + len(Traindata_v5) for i in range(len(Traindata_v5))]
                       ]

#print(train_fixed_exp_assignment)

scenario = ni_benchmark(
        torch.utils.data.ConcatDataset([Traindata,Traindata_v1,Traindata_v2,Traindata_v3,Traindata_v4,Traindata_v5]), 
        torch.utils.data.ConcatDataset([Evaldata,Evaldata_v1,Evaldata_v2,Evaldata_v3,Evaldata_v4,Evaldata_v5]), 
        6, task_labels=False, shuffle=False, fixed_exp_assignment=train_fixed_exp_assignment
    )
# MODEL CREATION
model = simpleNN()
#model.load_state_dict(torch.load("../model/SavedModels/modelTTfull"))

# DEFINE THE EVALUATION PLUGIN and LOGGERS
# The evaluation plugin manages the metrics computation.
# It takes as argument a list of metrics, collectes their results and returns
# them to the strategy it is attached to.

# log to Tensorboard
tb_logger = TensorboardLogger()

# log to text file
#text_logger = TextLogger(open('log.txt', 'a'))

# print to stdout
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=False),
    ROC_metrics(minibatch=True,epoch=True,experience=True,stream=False),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=False),
    forgetting_metrics(experience=True),
    timing_metrics(epoch=True, epoch_running=True),
    confusion_matrix_metrics(num_classes=2, save_image=True,
                             stream=True),
    loggers=[interactive_logger, tb_logger],
    benchmark = scenario
)

# CREATE THE STRATEGY INSTANCE (NAIVE)
# cl_strategy = Naive(
#     model, SGD(model.parameters(), lr=0.01, momentum=0.9),
#     BCELoss(), train_mb_size=500, train_epochs=4, eval_mb_size=100,
#     evaluator=eval_plugin,plugins=[ReplayP(mem_size=2000)])


cl_strategy = SynapticIntelligence(
        model,
        Adam(model.parameters(), lr=0.001),
        BCELoss(),
        si_lambda=0.0001,
        train_mb_size=128,
        train_epochs=4,
        eval_mb_size=128,
        evaluator=eval_plugin,
    )

# TRAINING LOOP
print('Starting experiment...')
results = []
for experience in scenario.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    # test also returns a dictionary which contains all the metric values
    results.append(cl_strategy.eval(scenario.test_stream))

# Save model
torch.save(model.state_dict(), "../model/SavedModels/modelTTfull_CL_SI")