from typing import List
from CL.Interface.Problem import Problem


class ProblemSequence:
    def __init__(self, problems: List[Problem]):
        self.problems = problems

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, i: int):
        return self.problems[i]
