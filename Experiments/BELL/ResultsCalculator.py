import numpy as np
from Experiments.Interface.SequenceAggregatedLogs import SequenceAggregatedLogs
from Experiments.Interface.ResultsCalculator import ResultsCalculator
from Experiments.BELL.Experiment import *


class BELLResultsCalculator(ResultsCalculator):
    @staticmethod
    def get_benchmark_name():
        return "BELL"

    @staticmethod
    def get_random_seeds(sequence: str):
        return experiments_dict[sequence]["random_seeds"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sequence", help=f"One of: {list(experiments_dict.keys())}", type=str)
    parser.add_argument("-cl_alg", help=f"One of: PICLE, MNTDP, random, HOUDINI, standalone, ewc, PICLE_PT, PICLE_NT", type=str)
    parser.add_argument("-dbg", help=f"If true, only the first random seed was evaluated", action='store_true')
    args = parser.parse_args()
    print(f"-sequence = {args.sequence}")
    print(f"-cl_alg = {args.cl_alg}")
    BELLResultsCalculator.get_alg_perf(args.sequence, args.cl_alg, args.dbg)
