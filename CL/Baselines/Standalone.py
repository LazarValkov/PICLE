import time
from CL.Interface.CLAlgorithm import CLAlg
from CL.Interface.Hypothesis import Hypothesis
from CL.Interface.ProblemSolvingLog import ProblemSolvingLog
from CL.Interface.Problem import Problem
from CL.Interface.NNEvaluator import NNEvaluator
from CL.Interface.NNTrainer import NNTrainer
import torch


class Standalone(CLAlg):
    def get_name(self):
        return "standalone"

    def solve_problem(self, problem: Problem, trainer: NNTrainer, max_time: float = None) \
            -> (Hypothesis, ProblemSolvingLog):
        c_random_seed = problem.num_id + self.random_seed if self.random_seed is not None else None

        assert trainer is not None
        evaluator = NNEvaluator(problem, self.device)

        sa_modules = (None,) * problem.architecture_class.get_num_modules()

        sa_model = problem.architecture_class(sa_modules, self.device,
                                              enable_finetuning=False, if_finetuning_finetune_a_copy=True,
                                              random_init_random_seed=c_random_seed,
                                              output_dim=problem.output_dim)

        start_time = time.time()
        trainer.train(sa_model, self.num_epochs)
        training_time = time.time() - start_time

        val_loss, val_acc = evaluator.evaluate_on_val_dataset(sa_model)
        test_loss, test_acc = evaluator.evaluate_on_test_dataset(sa_model)

        if self.device != "cpu":
            memory_used = torch.cuda.memory_allocated(device=self.device)
        else:
            memory_used = 0.

        problem_solving_log = ProblemSolvingLog(self.random_seed,
                                                is_alg_modular=False,
                                                time_taken=training_time,
                                                memory_used=memory_used,
                                                val_loss=val_loss, val_acc=val_acc,
                                                test_loss=test_loss, test_acc=test_acc)

        return sa_model, problem_solving_log

    def get_hypothesis_from_a_single_file(self, filepath: str, problem: Problem) -> Hypothesis:
        # note: not sure if this function was called during the experiments, so it might not have been tested
        num_modules = problem.architecture_class.get_num_modules()
        standalone_prog = tuple(None for _ in range(num_modules))
        hypothesis = problem.architecture_class(standalone_prog, self.device,
                                                enable_finetuning=False, if_finetuning_finetune_a_copy=True,
                                                output_dim=problem.output_dim,
                                                random_init_random_seed=None # not needed, no new parameters
                                                )
        hypothesis.load_modules_from_a_single_file(filepath, self.device)
        return hypothesis
