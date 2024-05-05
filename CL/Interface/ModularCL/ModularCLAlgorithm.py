"""
from LML.Algorithm import LML, Hypothesis, ProblemSolvingLog
from LML.NNArchitecture import NNArchitecture
from Experiments.Problem import Problem
from LML.NNEvaluator import NNEvaluator
from LML.NNTrainer import NNTrainer
from LML.ModularLML.Lib import Library
from LML.ModularLML.EvalCache import EvalCache
from LML.NNArchitecture import NNModule
"""
from abc import abstractmethod, ABC
from typing import Tuple, Union, List, Optional, Generator, Dict
import time
import pickle
import os
from pathlib import Path
import itertools
from random import Random
import torch

from CL.Interface.CLAlgorithm import *
from CL.Interface.NNModule import NNModule
from CL.Interface.NNArchitecture import NNArchitecture
from CL.Interface.NNTrainer import NNTrainer
from CL.Interface.NNEvaluator import NNEvaluator

from CL.Interface.ModularCL.Library import Library
from CL.Interface.ModularCL.EvalCache import EvalCache
from CL.Interface.ModularCL.EvaluatePathResult import EvaluatePathResult
from CL.Interface.ModularCL.ModularProblemSolvingLog import ModularProblemSolvingLog


class ModularCLAlg(CLAlg, ABC):
    def __init__(self,
                 random_seed: int,
                 device: str,
                 num_epochs: int,
                 lib_folder: str,
                 enable_finetuning=False,
                 if_finetuning_finetune_a_copy=True,
                 evaluations_cache_root_folder=None,
                 _debug_include_suggestion_time: bool = True,
                 _debug_dont_save_after_solving_problem: bool = False,
                 _debug_eval_cache_throw_exception_if_cant_load_module = True,
                 _debug_dont_evaluate_on_val_and_test = False,
                 _debug_evaluate_only_standalone_for_first_t_problems = None,
                 _debug_share_lib_folder=False
                 ):
        super().__init__(random_seed, device, num_epochs)

        self.lib_folder = lib_folder

        self.enable_finetuning = enable_finetuning
        self.if_finetuning_finetune_a_copy = if_finetuning_finetune_a_copy
        self.evaluations_cache_root_folder = evaluations_cache_root_folder
        self.use_evaluations_cache = self.evaluations_cache_root_folder is not None

        self._debug_include_suggestion_time = _debug_include_suggestion_time
        self._debug_dont_save_after_solving_problem = _debug_dont_save_after_solving_problem
        self._debug_eval_cache_throw_exception_if_cant_load_module = _debug_eval_cache_throw_exception_if_cant_load_module
        self._debug_dont_evaluate_on_val_and_test = _debug_dont_evaluate_on_val_and_test
        self._debug_evaluate_only_standalone_for_first_t_problems = _debug_evaluate_only_standalone_for_first_t_problems

        self.lib = None
        self._debug_share_lib_folder = _debug_share_lib_folder

    @abstractmethod
    def get_new_library(self, device) -> Library:
        pass

    def _get_model_from_program_strs(self,
                                     program_strs: Tuple[Union[str, None]],
                                     problem: Problem,
                                     trained_modules: Optional[List[NNModule]] = None,
                                     random_init_random_seed=None
                                     ) -> NNArchitecture:
        modules = []
        for m_str in program_strs:
            modules.append(None if m_str is None else self.lib[m_str].module)
        modules = tuple(modules)

        model = problem.architecture_class(modules, self.device,
                                            self.enable_finetuning, self.if_finetuning_finetune_a_copy,
                                           random_init_random_seed=random_init_random_seed,
                                           output_dim=problem.output_dim)

        if trained_modules is not None:
            trained_modules_dict = {tm.get_module_type_name(): tm for tm in trained_modules}
            for nm in model.trainable_modules:
                nm.load_state_dict(trained_modules_dict[nm.get_module_type_name()].state_dict())

        return model

    def explore_program(self, modules_str: Tuple[Union[str, None]], problem: Problem, trainer: NNTrainer,
                        evaluator: NNEvaluator,
                        eval_cache: Union[None, EvalCache],
                        random_init_random_seed=None) -> EvaluatePathResult:
        # note: there is a naming inconsistency. modules_str == program_strs elsewhere. (...sorry)

        # This is done, so that we can handle None by first converting it to "None"
        c_program_str = ' '.join((str(j) for j in modules_str))
        print(f"evaluating {modules_str}")

        model = self._get_model_from_program_strs(modules_str, problem, random_init_random_seed=random_init_random_seed)

        if eval_cache is not None and c_program_str in eval_cache.keys():
            c_cached_eval = eval_cache[c_program_str]

            exploration_time = c_cached_eval['exploration_time']
            val_loss = c_cached_eval['val_loss']
            val_acc = c_cached_eval['val_acc']
            learning_curves_dict = c_cached_eval['learning_curves_dict']

            eval_cache.load_newly_trained_modules(c_program_str, model)
        else:
            exploration_start = time.time()

            # Training if necessary
            if model.has_trainable_parameters():
                if random_init_random_seed is not None:
                    last_state = torch.get_rng_state()
                    torch.manual_seed(random_init_random_seed)
                _, learning_curves_dict = trainer.train(model, self.num_epochs)
                if random_init_random_seed is not None:
                    torch.set_rng_state(last_state)
            else:
                learning_curves_dict = {"tr_losses": [], "tr_accs": [], "val_losses": [], "val_accs": []}
            # Evaluate
            val_loss, val_acc = evaluator.evaluate_on_val_dataset(model)
            print(f"val_loss = {val_loss}")
            print(f"val_acc = {val_acc}")

            exploration_time = time.time() - exploration_start

            # update the cache
            if eval_cache is not None:
                eval_cache.add_evaluation(c_program_str, exploration_time, val_loss, val_acc,
                                          model.trainable_modules, learning_curves_dict)
                if not self._debug_dont_save_after_solving_problem:
                    eval_cache.save()   # save after every evaluation, so that debugging is easier.

        return EvaluatePathResult(c_program_str, modules_str, val_loss, val_acc,
                                  exploration_time, model, learning_curves_dict)

    def dbg_iterate_programs_sa_only(self, problem: Problem, trainer: NNTrainer, evaluator: NNEvaluator,
                         eval_cache: Union[EvalCache, None],
                         random_obj: Random, random_init_random_seed: int) -> Generator[EvaluatePathResult, None, None]:

        print("!!!! DBG: Iterating through the standalone program only!")
        eval_function = lambda modules_str: self.explore_program(modules_str, problem, trainer, evaluator,
                                                                 eval_cache, random_init_random_seed)
        # evaluate the standalone version
        c_num_module_layers = problem.architecture_class.get_num_modules()
        sa_program = (None,) * c_num_module_layers
        standalone_eval_res = eval_function(sa_program)
        yield standalone_eval_res

    @abstractmethod
    def iterate_programs(self, problem: Problem, trainer: NNTrainer, evaluator: NNEvaluator,
                         eval_cache: Union[EvalCache, None],
                         random_obj: Random, random_init_random_seed: int) -> Generator[EvaluatePathResult, None, None]:
        """
        yield program_str, loss, acc, exploration_time, newly_trained_modules
        """
        pass

    def solve_problem(self, problem: Problem,
                      trainer: NNTrainer, max_time: float = None,
                      _dbg_max_num_programs=None,
                      _dbg_lib: Optional[Library] = None,
                      _dbg_return_best_program_str_tuple_as_well=False
                      ) -> (Hypothesis, ModularProblemSolvingLog):
        """
        Search for the best program, given the constraints
        Note: use the validation inputs and validation labels only for logging the search performances.
        """
        if _dbg_lib is not None:
            self.lib = _dbg_lib
        else:
            self.lib = self.get_new_library(self.device)
            if problem.num_id > 0:
                # load the library saved after solving the previous problem

                if self._debug_share_lib_folder:
                    prev_lib_folder = f"{self.lib_folder}/"
                else:
                    prev_lib_folder = f"{self.lib_folder}/{problem.num_id - 1}/"
                self.lib.load(prev_lib_folder, problem.num_id - 1)

        if _dbg_max_num_programs is not None:
            assert max_time is None

        L = problem.architecture_class.get_num_modules()
        num_solved_problems = problem.num_id
        c_max_num_paths_to_train = 2*L + num_solved_problems -1 + 1 # (+1 for the standalone model)

        if self.get_should_limit_evaluations_by_number_of_paths() and _dbg_max_num_programs is None:
            _dbg_max_num_programs = c_max_num_paths_to_train
            print(f"Gonna evaluate only {_dbg_max_num_programs} different paths.")

        c_random_seed = problem.num_id + self.random_seed if self.random_seed is not None else None
        random_obj = Random(c_random_seed)

        if self.use_evaluations_cache:
            c_eval_folder = f"{self.evaluations_cache_root_folder}/{problem.num_id}_{problem.name}/"
            eval_cache = EvalCache(c_eval_folder, self._debug_eval_cache_throw_exception_if_cant_load_module)
        else:
            eval_cache = None

        evaluator = NNEvaluator(problem, self.device)

        programs, times, \
            val_losses, val_accuracies, \
            test_losses, test_accuracies, \
            per_program_learning_plot_dicts = [], [], [], [], [], [], []

        total_time_taken = 0.

        # calculated using the val dataset
        c_best_program_str = c_best_model = None
        c_best_program_strs_tuple = None
        c_best_val_loss = c_best_val_acc = None
        c_best_test_loss = c_best_test_acc = None

        memory_taken_logs = []

        if self._debug_evaluate_only_standalone_for_first_t_problems is not None \
                and self._debug_evaluate_only_standalone_for_first_t_problems > problem.num_id:
            c_program_iterating_function = self.dbg_iterate_programs_sa_only
        else:
            c_program_iterating_function = self.iterate_programs

        for c_res in c_program_iterating_function(problem, trainer, evaluator, eval_cache, random_obj, c_random_seed):
            # standalone was just evaluated
            if len(programs) == 0 and self.get_should_limit_evaluations_by_time() and max_time is None:
                max_time = c_res.exploration_time * (c_max_num_paths_to_train - 1)  # -1 cos we already evaluated SA
                print(f"limiting the total time to {max_time} seconds")

            if not self._debug_dont_evaluate_on_val_and_test:
                new_val_loss, new_val_acc = evaluator.evaluate_on_val_dataset(c_res.model)
                new_test_loss, new_test_acc = evaluator.evaluate_on_test_dataset(c_res.model)
            else:
                new_val_loss, new_val_acc = c_res.val_loss, c_res.val_acc
                new_test_loss = new_test_acc = -1

            memory_taken_logs.append(torch.cuda.memory_allocated(device=self.device))

            total_time_taken += c_res.exploration_time

            programs.append(c_res.program_str)
            times.append(total_time_taken)
            val_losses.append(new_val_loss)
            val_accuracies.append(new_val_acc)
            test_losses.append(new_test_loss)
            test_accuracies.append(new_test_acc)
            per_program_learning_plot_dicts.append(c_res.learning_curves_dict)

            if c_best_val_acc is None or c_best_val_acc < c_res.val_acc:
                c_best_program_str = c_res.program_str
                c_best_program_strs_tuple = c_res.program_strs_tuple
                c_best_model = c_res.model

                c_best_val_loss, c_best_val_acc = new_val_loss, new_val_acc
                c_best_test_loss, c_best_test_acc = new_test_loss, new_test_acc

            if max_time is not None and total_time_taken >= max_time:
                break

            if _dbg_max_num_programs is not None and len(programs) >= _dbg_max_num_programs:
                break

        min_val_losses = [min(val_losses[:l]) for l in range(1, len(val_losses) + 1)]
        max_val_accs = [max(val_accuracies[:l]) for l in range(1, len(val_accuracies) + 1)]

        min_test_losses = [min(test_losses[:l]) for l in range(1, len(test_losses) + 1)]
        max_test_accs = [max(test_accuracies[:l]) for l in range(1, len(test_accuracies) + 1)]

        if not self._debug_dont_save_after_solving_problem:
            # update the library
            self.update_lib(problem, c_best_model, c_best_val_acc)
            c_best_program_strs_tuple_in_updated_lib = []
            for m_idx, m in enumerate(c_best_program_strs_tuple):
                if m is not None:
                    c_best_program_strs_tuple_in_updated_lib.append(m)
                else:
                    new_m = f"{problem.num_id}_{problem.architecture_class.get_module_types()[m_idx].get_module_type_name()}"
                    c_best_program_strs_tuple_in_updated_lib.append(new_m)
            c_best_program_strs_tuple_in_updated_lib = tuple(c_best_program_strs_tuple_in_updated_lib)
            self.lib.previous_solutions.append(c_best_program_strs_tuple_in_updated_lib)

            # save the library
            assert problem.num_id is not None
            assert self.lib_folder is not None

            if self._debug_share_lib_folder:
                new_lib_folder = f"{self.lib_folder}/"
            else:
                new_lib_folder = f"{self.lib_folder}/{problem.num_id}/"
            self.lib.save(new_lib_folder, problem.num_id, save_all_modules=not self._debug_share_lib_folder)

            # save the cache
            if eval_cache is not None:
                eval_cache.save()

        max_memory_used = max(memory_taken_logs)
        problem_solving_log = ModularProblemSolvingLog(c_best_program_str, programs, times,
                                                       val_losses, val_accuracies,
                                                       test_losses, test_accuracies,
                                                       min_val_losses, max_val_accs,
                                                       min_test_losses, max_test_accs,
                                                       per_program_learning_plot_dicts,

                                                       random_seed=self.random_seed,
                                                       is_alg_modular=True,
                                                       time_taken=total_time_taken,
                                                       memory_used=max_memory_used,
                                                       val_loss=c_best_val_loss,
                                                       val_acc=c_best_val_acc,
                                                       test_loss=c_best_test_loss,
                                                       test_acc=c_best_test_acc)
        if _dbg_return_best_program_str_tuple_as_well:  # used for MNTDP
            return c_best_model, problem_solving_log, c_best_program_strs_tuple

        return c_best_model, problem_solving_log

    def update_lib(self, problem: Problem, final_model: NNArchitecture, performance):
        # add the most successful modules to the library
        assert self.enable_finetuning is False  # remove after you see below concerns
        # TODO: If finetuning is enabled, I need to double-check the following for the propert behaviour:
        # This might cause errors if I allow modules to be finetuned instead of freezing.
        # Also if a copy is finetuned, I need to add the copy to the library.
        # Consider using final_model.new_modules
        for c_module in final_model.trainable_modules:
            c_module_name = f"{problem.num_id}_{c_module.get_module_type_name()}"
            self.lib.add_module(c_module_name, c_module, tr_inputs=None, tr_outputs=None, performance=performance)

    def get_hypothesis_from_a_single_file(self, filepath: str, problem: Problem) -> Hypothesis:
        num_modules = problem.architecture_class.get_num_modules()
        standalone_prog = tuple(None for _ in range(num_modules))
        hypothesis = problem.architecture_class(standalone_prog,
                                                self.device, self.enable_finetuning, self.if_finetuning_finetune_a_copy,
                                                output_dim=problem.output_dim)
        hypothesis.load_modules_from_a_single_file(filepath, self.device)
        return hypothesis

    def get_should_limit_evaluations_by_time(self):
        return False

    def get_should_limit_evaluations_by_number_of_paths(self):
        return False
