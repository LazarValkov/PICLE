from typing import List, Dict
from CL.Interface.ProblemSolvingLog import ProblemSolvingLog


class ModularProblemSolvingLog(ProblemSolvingLog):
    def __init__(self,
                 best_prog_found: str,
                 programs: List[str],
                 times: List[float],

                 val_losses: List[float],
                 val_accuracies: List[float],
                 test_losses: List[float],
                 test_accuracies: List[float],

                 min_val_losses: List[float],
                 max_val_accuracies: List[float],
                 min_test_losses: List[float],
                 max_test_accuracies: List[float],

                 per_program_learning_plot_dicts: List[Dict[str, List[float]]],

                 *args, **kwargs):
        self.best_prog_found = best_prog_found
        self.programs = programs
        self.times = times

        self.val_accuracies = val_accuracies
        self.val_losses = val_losses
        self.test_accuracies = test_accuracies
        self.test_losses = test_losses

        self.min_val_losses = min_val_losses
        self.max_val_accuracies = max_val_accuracies
        self.min_test_losses = min_test_losses
        self.max_test_accuracies = max_test_accuracies

        self.per_program_learning_plot_dicts = per_program_learning_plot_dicts

        super().__init__(*args, **kwargs)