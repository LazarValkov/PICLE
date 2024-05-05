from typing import Dict, List
from Experiments.Interface.ProblemAggregatedLogs import ProblemAggregatedLogs
from CL.Interface.ProblemSolvingLog import ProblemSolvingLog
from CL.Interface.Problem import Problem
from CL.Interface.CLAlgorithm import CLAlg
import json


class SequenceAggregatedLogs:
    def __init__(self, folder, dbg_filepath=None):
        if dbg_filepath is not None:
            self.filepath = dbg_filepath
        else:
            self.filepath = f"{folder}/aggregated_logs.json"
        self.problem_aggregated_logs: List[ProblemAggregatedLogs] = []

    def add_log(self, problem: Problem, alg: CLAlg, log: ProblemSolvingLog):
        assert problem.num_id <= len(self.problem_aggregated_logs)
        if problem.num_id == len(self.problem_aggregated_logs):
            self.problem_aggregated_logs.append(ProblemAggregatedLogs(problem.num_id, problem.name))
        self.problem_aggregated_logs[problem.num_id].add_alg_log(alg.get_name(), log)

    def _get_dict(self) -> Dict:
        return {
            'problem_aggregated_logs': [pal._get_dict() for pal in self.problem_aggregated_logs]
        }

    def load_from_dict(self, c_dict: Dict):
        for c_pal in c_dict['problem_aggregated_logs']:
            c_problem_aggregated_logs = ProblemAggregatedLogs.create_from_dict(c_pal)
            self.problem_aggregated_logs.append(c_problem_aggregated_logs)

    def save_to_file(self):
        c_dict = self._get_dict()
        with open(self.filepath, 'w') as outfile:
            json.dump(c_dict, outfile)

    def load_from_file(self):
        with open(self.filepath) as json_file:
            c_dict = json.load(json_file)
        self.load_from_dict(c_dict)

    def remove_alg(self, alg_name):
        for p in self.problem_aggregated_logs:
            p.remove_alg(alg_name)
