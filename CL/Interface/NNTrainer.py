from collections import OrderedDict
from abc import ABC, abstractmethod
from CL.Interface.Problem import Problem
from CL.Interface.NNArchitecture import NNArchitecture


def _clone_hidden_state(state):
    result = OrderedDict()
    for key, val in state.items():
        result[key] = val.clone()
    return result


class NNTrainer(ABC):
    def __init__(self, problem: Problem, device: str, learning_rate, weight_decay):
        self.problem = problem
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion_class = problem.architecture_class.get_criterion_class()


    @abstractmethod
    def train(self, model: NNArchitecture, num_epochs: int, additional_loss_fn=None):
        """
        Trains the given model for num_epochs epochs
        :param additional_loss_fn: can be used for regularization
        """
        pass
