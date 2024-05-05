import os
from pathlib import Path
from typing import Union
import fcntl
from abc import ABC, abstractmethod
from CL.Interface.CLAlgorithm import CLAlg
from CL.Interface.ProblemSequence import ProblemSequence
from CL.Interface.Problem import Problem
from CL.Interface.Hypothesis import Hypothesis
from CL.Interface.ProblemSolvingLog import ProblemSolvingLog
from CL.Interface.NNTrainer import NNTrainer
from Experiments.Interface.SequenceAggregatedLogs import SequenceAggregatedLogs


class Experiment(ABC):
    def __init__(self, max_time: Union[None, float], device: str,
                 alg: CLAlg, problem_seq: ProblemSequence,
                 root_result_folder: str,
                 save_hypothesis=False):
        self.device = device
        self.max_time = max_time

        self.alg = alg
        self.problem_seq = problem_seq

        self.root_result_folder = root_result_folder
        self.alg_subfolder = f"{self.root_result_folder}/{alg.get_name()}/{alg.random_seed}"
        os.makedirs(self.alg_subfolder, exist_ok=True)

        self.save_hypothesis = save_hypothesis

        self.lock_filepath = f"{root_result_folder}/lock.lock"
        if not os.path.exists(self.lock_filepath):
            Path(self.lock_filepath).touch()

    def save_results(self, problem: Problem, hypothesis: Hypothesis, log: ProblemSolvingLog):
        if self.save_hypothesis:
            # save the hypothesis
            h_filepath = f"{self.alg_subfolder}/hypothesis_{problem.num_id}.h"
            hypothesis.save_to_a_single_file(h_filepath)

        # Using the locks prevents another process
        # from saving something else after we load the file until we save.
        with Locker(self.lock_filepath):
            aggregated_logs = SequenceAggregatedLogs(self.root_result_folder)
            if os.path.exists(aggregated_logs.filepath):
                aggregated_logs.load_from_file()
            aggregated_logs.add_log(problem, self.alg, log)
            aggregated_logs.save_to_file()

    def run_sequence(self, start_from_idx=0, max_num_programs=None, max_num_problems=None):
        for c_prob_idx, c_problem in enumerate(self.problem_seq):
            if c_prob_idx < start_from_idx:
                continue
            print(f"Solving {c_problem.num_id}. {c_problem.name} using {self.alg.get_name()}")
            nntrainer = self.get_nn_trainer(c_problem)
            if max_num_programs is not None:
                h, log = self.alg.solve_problem(c_problem, nntrainer, self.max_time,
                                                _dbg_max_num_programs=max_num_programs)
            else:
                h, log = self.alg.solve_problem(c_problem, nntrainer, self.max_time)

            self.save_results(c_problem, h, log)
            h = None  # release from memory. not gonna save it atm
            c_problem.unload_datasets()
            print(f"Final val acc = {log.val_acc}, final test acc = {log.test_acc}")

            if (max_num_problems is not None) and (c_prob_idx+1 == max_num_problems):
                break

    @abstractmethod
    def get_nn_trainer(self, problem) -> NNTrainer:
        pass


class Locker:
    def __init__(self, lock_filepath):
        self.lock_filepath = lock_filepath

    def __enter__ (self):
        self.fp = open(self.lock_filepath)
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__ (self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()