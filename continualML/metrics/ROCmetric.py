from typing import List, Union, Dict

from avalanche.evaluation import GenericPluginMetric,PluginMetric
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name
from avalanche.evaluation import Metric

import torch
from torch import Tensor
import sklearn.metrics as metrics


# a standalone metric implementation
class ROC(Metric[float]):
    """
    This metric will return a `float` value
    """
    def __init__(self):
        """
        Initialize your metric here
        """
        super().__init__()
        self.y_array = []
        self.y_true_array = []
        pass

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor,) -> None:
        """Update the running arrays given the true and predicted labels.
        :param predicted_y: The model prediction.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :return: None.
        """
        self.y_array.append(predicted_y.to_numpy())
        self.y_true_array.append(true_y.to_numpy())

    def result(self) -> float:
        return metrics.roc_auc_score(self.y_true_array,self.y_array)

    def reset(self):
        self.y_array = []
        self.y_true_array = []

class ROCPluginMetric(GenericPluginMetric[float]):
    """
    This metric will return a `float` value after
    each training epoch
    """

    def __init__(self):
        """
        Initialize the metric
        """
        super().__init__()

        self._ROC = ROC()

    def __init__(self, reset_at, emit_at, mode):
        """Creates the ROC plugin
        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware ROC or not.
        """

        self._ROC = ROC()
        super(ROCPluginMetric, self).__init__(
            self._ROC, reset_at=reset_at, emit_at=emit_at, mode=mode
        )

    def reset(self) -> None:
        """
        Reset the metric
        """
        self._ROC.reset()

    def result(self) -> float:
        """
        Emit the result
        """
        return self._ROC.result()

    def after_training_iteration(self, strategy: 'PluggableStrategy') -> None:
        """
        Update the ROC metric with the current
        predictions and targets
        """
        # task labels defined for each experience
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]

        self._ROC.update(strategy.mb_output, strategy.mb_y, 
                                     task_labels)


    def before_training_epoch(self, strategy: 'PluggableStrategy') -> None:
        """
        Reset the accuracy before the epoch begins
        """
        self.reset()

    def after_training_epoch(self, strategy: 'PluggableStrategy'):
        """
        Emit the result
        """
        return self._package_result(strategy)
        
        
    def _package_result(self, strategy):
        """Taken from `GenericPluginMetric`, check that class out!"""
        metric_value = self.accuracy_metric.result()
        add_exp = False
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=add_exp, add_task=k)
                metrics.append(MetricValue(self, metric_name, v,
                                           plot_x_position))
            return metrics
        else:
            metric_name = get_metric_name(self, strategy,
                                          add_experience=add_exp,
                                          add_task=True)
            return [MetricValue(self, metric_name, metric_value,
                                plot_x_position)]

    def __str__(self):
        """
        Here you can specify the name of your metric
        """
        return "Top1_ROC_Epoch"

class MinibatchROC(ROCPluginMetric):
    """
    The minibatch plugin ROC metric.
    This metric only works at training time.
    This metric computes the average ROC over patterns
    from a single minibatch.
    It reports the result after each iteration.
    If a more coarse-grained logging is needed, consider using
    :class:`EpochROC` instead.
    """

    def __init__(self):
        """
        Creates an instance of the MinibatchROC metric.
        """
        super(MinibatchROC, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "Top1_ROC_MB"

class EpochROC(ROCPluginMetric):
    """
    The average ROC over a single training epoch.
    This plugin metric only works at training time.
    The ROC will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochROC metric.
        """

        super(EpochROC, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train"
        )

    def __str__(self):
        return "Top1_ROC_Epoch"

class RunningEpochROC(ROCPluginMetric):
    """
    The average ROC across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.
    At each iteration, this metric logs the ROC averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningEpochROC metric.
        """

        super(RunningEpochROC, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "Top1_RunningROC_Epoch"

class ExperienceROC(ROCPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average ROC over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceROC metric
        """
        super(ExperienceROC, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "Top1_ROC_Exp"

class StreamROC(ROCPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average ROC over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamROC metric
        """
        super(StreamROC, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "Top1_ROC_Stream"

class TrainedExperienceROC(ROCPluginMetric):
    """
    At the end of each experience, this plugin metric reports the average
    ROC for only the experiences that the model has been trained on so far.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of TrainedExperienceROC metric by first
        constructing ROCPluginMetric
        """
        super(TrainedExperienceROC, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )
        self._current_experience = 0

    def after_training_exp(self, strategy) -> None:
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience
        ROCPluginMetric.reset(self, strategy)
        return ROCPluginMetric.after_training_exp(self, strategy)

    def update(self, strategy):
        """
        Only update the ROC with results from experiences that have been
        trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            ROCPluginMetric.update(self, strategy)

    def __str__(self):
        return "ROC_On_Trained_Experiences"


def ROC_metrics(
    *,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    stream=False,
    trained_experience=False,
) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.
    :param minibatch: If True, will return a metric able to log
        the minibatch ROC at training time.
    :param epoch: If True, will return a metric able to log
        the epoch ROC at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch ROC at training time.
    :param experience: If True, will return a metric able to log
        the ROC on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the ROC averaged over the entire evaluation stream of experiences.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation ROC only for experiences that the
        model has been trained on
    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchROC())

    if epoch:
        metrics.append(EpochROC())

    if epoch_running:
        metrics.append(RunningEpochROC())

    if experience:
        metrics.append(ExperienceROC())

    if stream:
        metrics.append(StreamROC())

    if trained_experience:
        metrics.append(TrainedExperienceROC())

    return metrics


__all__ = [
    "ROC",
    "MinibatchROC",
    "EpochROC",
    "RunningEpochROC",
    "ExperienceROC",
    "StreamROC",
    "TrainedExperienceROC",
    "ROC_metrics",
]