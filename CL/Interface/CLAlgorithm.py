from abc import ABC, abstractmethod
from CL.Interface.Problem import Problem
from CL.Interface.Hypothesis import Hypothesis
from CL.Interface.ProblemSolvingLog import ProblemSolvingLog
from CL.Interface.NNTrainer import NNTrainer


class CLAlg(ABC):
    def __init__(self, random_seed: int, device: str, num_epochs: int):
        self.random_seed = random_seed
        self.device = device
        self.num_epochs = num_epochs

    @abstractmethod
    def get_name(self):
        # return the name of the CL algorithm
        pass

    @abstractmethod
    def solve_problem(self, problem: Problem, trainer: NNTrainer, max_time: float = None) \
            -> (Hypothesis, ProblemSolvingLog):
        pass

    @abstractmethod
    def get_hypothesis_from_a_single_file(self, filepath: str, problem: Problem) -> Hypothesis:
        pass
