import torch
torch.use_deterministic_algorithms(True)
import numpy as np
from Experiments.Interface.ProblemAggregatedLogs import ProblemAggregatedLogs
from Experiments.Interface.SequenceAggregatedLogs import SequenceAggregatedLogs
from abc import ABC, abstractmethod


def _get_all_not_None(items: list) -> bool:
    for i in items:
        if i is None:
            return False
    return True


class ResultsCalculator(ABC):
    @staticmethod
    def get_standalone_perf(c_pal: ProblemAggregatedLogs):
        available_algs = c_pal.algorithm_to_logs_list.keys()
        if "standalone" in available_algs:
            return c_pal.algorithm_to_logs_list["standalone"][-1].test_acc
        if "MNTDP" in available_algs:
            return c_pal.algorithm_to_logs_list["MNTDP"][-1].test_accuracies[-1]
        if "PICLE" in available_algs:
            return c_pal.algorithm_to_logs_list["PICLE"][-1].test_accuracies[0]
        # can't calculate the standalone performance
        return None

    @staticmethod
    def get_alg_perf_for_given_problem(c_pal: ProblemAggregatedLogs,
                                       c_alg_name: str):
        standalone_acc = ResultsCalculator.get_standalone_perf(c_pal)
        c_alg_test_acc = c_pal.algorithm_to_logs_list[c_alg_name][-1].test_acc
        c_alg_test_acc_transfer = c_alg_test_acc - standalone_acc if standalone_acc is not None else None
        return c_alg_test_acc, c_alg_test_acc_transfer

    @staticmethod
    def get_alg_perfs_for_given_sequence(c_aggregated_logs: SequenceAggregatedLogs,
                                         alg_name: str,
                                         _dbg_first_N_problems=None):
        # Computes the performance of the given algorithm name on the given sequence
        accuracies = []
        transfers = []
        # Iterate through each problem and get the accuracy
        for i in range(len(c_aggregated_logs.problem_aggregated_logs)):
            if _dbg_first_N_problems is not None and i == _dbg_first_N_problems:
                break
            c_pal = c_aggregated_logs.problem_aggregated_logs[i]
            c_acc, c_tr = ResultsCalculator.get_alg_perf_for_given_problem(c_pal, alg_name)
            accuracies.append(c_acc)
            transfers.append(c_tr)

        mean_acc = np.array(accuracies).mean()
        # mean_tr = np.array(transfers).mean()
        last_transfer = transfers[-1]
        # print("***************************")
        # print(f"Accuracies: {accuracies}")
        return mean_acc, last_transfer

    @staticmethod
    @abstractmethod
    def get_random_seeds(sequence: str):
        pass

    @staticmethod
    @abstractmethod
    def get_benchmark_name():
        pass

    @classmethod
    def get_alg_perf(cls, sequence: str, alg_name: str,
                     _dbg,
                     _dbg_first_N_problems=None, _dbg_specific_random_seeds=None):
        # Computes the performances of the given algorithm name on the given sequence,
        # averaged across different random seeds

        mean_accuracies_for_each_seed = []
        last_transfers_for_each_seed = []

        random_seeds = cls.get_random_seeds(sequence)
        for rs in random_seeds:
            if _dbg_specific_random_seeds is not None and rs not in _dbg_specific_random_seeds:
                print(f"skipping random seed = {rs}")
            results_folder = f"results/{cls.get_benchmark_name()}/{sequence}/rs_{rs}/Results"
            print(f"loading {results_folder}")

            c_aggregated_logs = SequenceAggregatedLogs(results_folder)
            c_aggregated_logs.load_from_file()

            mean_acc, last_transfer = cls.get_alg_perfs_for_given_sequence(c_aggregated_logs,
                                                                           alg_name, _dbg_first_N_problems)
            mean_accuracies_for_each_seed.append(mean_acc)
            last_transfers_for_each_seed.append(last_transfer)

            if _dbg:
                break

        # print()
        print(f"mean_accuracies_for_each_seed={mean_accuracies_for_each_seed}")
        print(f"last_transfers_for_each_seed={last_transfers_for_each_seed}")

        avg_mean_accuracy_over_all_seeds = np.array(mean_accuracies_for_each_seed).mean()
        if _get_all_not_None(last_transfers_for_each_seed):
            avg_last_transfer_over_all_seeds = np.array(last_transfers_for_each_seed).mean()
        else:
            print("Some of the transfers couldn't be computed. Check that a standalone baseline was run.")
            avg_last_transfer_over_all_seeds = None

        return avg_mean_accuracy_over_all_seeds, avg_last_transfer_over_all_seeds