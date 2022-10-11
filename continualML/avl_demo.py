
from torch.optim import SGD
from torch.nn import CrossEntropyLoss, BCELoss
from dataset.torch_dataset import TrackDataset, Randomiser
from torch.utils.data import DataLoader
from avalanche.benchmarks import ni_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive

from torchvision.datasets import MNIST
from model.simpleNN import simpleNN
import torch


Traindata = TrackDataset("../dataset/Train/train.pkl")
Testdata = TrackDataset("../dataset/Test/test.pkl")

scenario = ni_benchmark(
        Traindata, Testdata, 5, task_labels=True, seed=1234
    )


# MODEL CREATION
model = simpleNN()
#model.load_state_dict(torch.load("../model/SavedModels/simplemodel"))

# DEFINE THE EVALUATION PLUGIN and LOGGERS
# The evaluation plugin manages the metrics computation.
# It takes as argument a list of metrics, collectes their results and returns
# them to the strategy it is attached to.

# log to Tensorboard
tb_logger = TensorboardLogger()

# log to text file
text_logger = TextLogger(open('log.txt', 'a'))

# print to stdout
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    forgetting_metrics(experience=True, stream=True),
    cpu_usage_metrics(experience=True),
    confusion_matrix_metrics(num_classes=1, save_image=False,
                             stream=True),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger, text_logger, tb_logger],
    benchmark = scenario
)

# CREATE THE STRATEGY INSTANCE (NAIVE)
cl_strategy = Naive(
    model, SGD(model.parameters(), lr=0.01, momentum=0.9),
    BCELoss(), train_mb_size=500, train_epochs=4, eval_mb_size=100,
    evaluator=eval_plugin)

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
torch.save(model.state_dict(), "../model/SavedModels/simplemodel_CL")