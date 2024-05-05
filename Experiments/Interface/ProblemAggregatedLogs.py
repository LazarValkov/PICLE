from typing import Dict, List
from CL.Interface.ProblemSolvingLog import ProblemSolvingLog
from CL.Interface.ModularCL.ModularProblemSolvingLog import ModularProblemSolvingLog


class ProblemAggregatedLogs:
    def __init__(self, problem_id, problem_name):
        self.problem_id = problem_id
        self.problem_name = problem_name
        self.algorithm_to_logs_list: Dict[str, List[ProblemSolvingLog]] = {}

    def add_alg_log(self, alg_name, log: ProblemSolvingLog):
        if alg_name not in self.algorithm_to_logs_list.keys():
            self.algorithm_to_logs_list[alg_name] = []
        self.algorithm_to_logs_list[alg_name].append(log)

    def _get_dict(self) -> Dict:
        return {
            'problem_id': self.problem_id,
            'problem_name': self.problem_name,
            'algorithm_to_logs_list': {alg_name: [psl.__dict__ for psl in list_psl]
                                       for alg_name, list_psl in self.algorithm_to_logs_list.items()}
        }

    def remove_alg(self, alg_name):
        if alg_name in self.algorithm_to_logs_list.keys():
            del self.algorithm_to_logs_list[alg_name]

    @staticmethod
    def create_from_dict(c_dict):
        res = ProblemAggregatedLogs(c_dict['problem_id'], c_dict['problem_name'])
        for alg, list_psl in c_dict['algorithm_to_logs_list'].items():
            res.algorithm_to_logs_list[alg] = [
                ModularProblemSolvingLog(**psl) if psl["is_alg_modular"] else ProblemSolvingLog(**psl)
                for psl in list_psl]
        return res
