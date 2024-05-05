import random
import numpy as np
from typing import List, Tuple, Dict, Union, Callable
import copy
from abc import ABC, abstractmethod
from CL.Interface.Problem import Problem
from CL.PICLE.Library import PICLELibrary
from CL.Interface.NNModule import NNModule
from CL.PICLE.NTSearch_Surrogate import GPModel
from CL.PICLE.NTSearch_FnDistance import FunctionalDistanceCalculator
from CL.Interface.ModularCL.EvaluatePathResult import EvaluatePathResult


class PICLE_NTPathSelector(ABC):
    def __init__(self,
                 problem: Problem,
                 lib: PICLELibrary,
                 partial_paths_to_search_through: List[Union[Tuple, Tuple[str]]]
                 ):
        self.problem = problem
        self.lib = lib
        self.partial_paths_to_search_through = partial_paths_to_search_through
        self.index_of_best_path = None

    @abstractmethod
    def suggest_next_program(self):
        pass

    @abstractmethod
    def evaluate_suggested_program(self, eval_function: Callable[[str], EvaluatePathResult]) -> EvaluatePathResult:
        pass


class PICLE_NTPathSelector_BOpt(PICLE_NTPathSelector):
    def __init__(self, problem: Problem,
                 lib: PICLELibrary,
                 partial_paths_to_search_through: List[Union[Tuple, Tuple[str]]],
                 standalone_prog_trained_modules,
                 standalone_prog_performance,
                 device,
                 #dbg_top_down_random_init=False,
                 #dbg_top_down_random_init_seed=None,
                 num_random_init=2,
                 EI_epsilon=0.001,  # 0.0005 works as well
                 ):
        super().__init__(problem, lib, partial_paths_to_search_through)
        self.num_random_init = num_random_init
        self.EI_epsilon = EI_epsilon
        # self.kernels = kernels
        #self.dbg_top_down_random_init = dbg_top_down_random_init
        #self.dbg_top_down_random_init_seed = dbg_top_down_random_init_seed

        # self.standalone_performance = standalone_prog_performance
        self.len_of_partial_paths = len(partial_paths_to_search_through[0])

        self.dict_partial_prog_strs_to_idx = {partial_prog: i for i, partial_prog in
                                              enumerate(partial_paths_to_search_through)}
        self.dict_partial_prog_strs_to_idx[(None,) * self.len_of_partial_paths] = len(partial_paths_to_search_through)
        self.dict_idx_to_partial_prog_strs = {v: k for k, v in self.dict_partial_prog_strs_to_idx.items()}

        partial_path_first_module_layer_idx = problem.architecture_class.get_num_modules() - self.len_of_partial_paths

        # sa_pprog = standalone_prog_trained_modules[-self.len_of_partial_paths:]
        # sa_pprog_strs = self.dict_idx_to_partial_prog_strs[len(partial_paths_to_search_through)]

        self.fn_distance_calculator = FunctionalDistanceCalculator(lib,
                                                                   self.problem.architecture_class,
                                                                   partial_path_first_module_layer_idx,
                                                                   device, distance_measure="l2")

        # cache the outputs of the standalone program
        self.fn_distance_calculator.cache_partial_program(standalone_prog_trained_modules[-self.len_of_partial_paths:],
                                                          self.dict_idx_to_partial_prog_strs[
                                                              len(partial_paths_to_search_through)])

        self.fn_distances_matrix = self.get_fn_distances_matrix(
            self.fn_distance_calculator.get_fn_distance_between_two_partial_programs)
        self.init_prog_indices_to_eval = self.get_initial_prog_indices_to_evaluate()

        # surrogate-related properties
        self.surrogate = None

        # the initial training indices are the standalone program
        self.meta_ds_train_indices = np.array([len(partial_paths_to_search_through)], dtype=np.int32).reshape((-1, 1))
        self.meta_ds_train_out = np.array([standalone_prog_performance], dtype=np.float32).reshape((-1, 1))
        self.meta_ds_test_indices = np.array(range(len(partial_paths_to_search_through)), dtype=np.int32).reshape((-1, 1))

        self.last_suggestion_prog_idx = None
        self.last_suggestion_idx_in_meta_ds_test = None
        self.last_suggestion_partial_prog_strs = None

    def get_initial_prog_indices_to_evaluate(self):
        # return the programs with the minimal average distances to others since these should be the most informative
        minimum_l2_distances = np.argsort(self.fn_distances_matrix[:-1].mean(axis=1))
        return minimum_l2_distances[:self.num_random_init].tolist()

    def get_fn_distances_matrix(self, calculating_fn):
        num_all_programs_involved = len(self.dict_partial_prog_strs_to_idx.keys())
        matrix = np.ones((num_all_programs_involved, num_all_programs_involved), dtype=np.float32)

        for i in range(num_all_programs_involved):
            for j in range(i, num_all_programs_involved):
                pp1 = self.dict_idx_to_partial_prog_strs[i]
                pp2 = self.dict_idx_to_partial_prog_strs[j]

                similarity = calculating_fn(pp1, pp2)
                matrix[i, j] = similarity
                matrix[j, i] = similarity
        return matrix

    def suggest_next_program(self):
        # First, check if we need to iterate the initial programs first
        if len(self.init_prog_indices_to_eval) > 0:
            new_suggestion_prog_idx = self.init_prog_indices_to_eval.pop(0)
            new_suggestion_program_strs = self.dict_idx_to_partial_prog_strs[new_suggestion_prog_idx]
            new_suggestion_aq_value = float("-inf")

            self.last_suggestion_prog_idx = new_suggestion_prog_idx
            self.last_suggestion_idx_in_meta_ds_test = np.where(self.meta_ds_test_indices.reshape((-1,)) == new_suggestion_prog_idx)
            self.last_suggestion_partial_prog_strs = new_suggestion_program_strs
            return new_suggestion_program_strs, new_suggestion_aq_value

        if self.meta_ds_test_indices.shape[0] == 0:
            return None

        self.surrogate = GPModel(self.fn_distances_matrix)

        self.surrogate.update_model(self.meta_ds_train_indices, self.meta_ds_train_out)

        new_suggestion_idx_in_meta_ds_test, aq_value_EI, aq_value_UCB = self.surrogate.select_next_index(
            self.meta_ds_test_indices)
        new_suggestion_prog_idx = self.meta_ds_test_indices[new_suggestion_idx_in_meta_ds_test].item()
        new_suggestion_program_strs = self.dict_idx_to_partial_prog_strs[new_suggestion_prog_idx]

        # print(f"Suggesting {new_suggestion_program_strs}, EI={aq_value_EI}, LCB={aq_value_UCB}, c_min={np.min(self.meta_ds_train_out)}")

        # if the Expected Improvement is almost 0 (up to some epsilon), return None
        # since we are not interested in evaluating the rest
        if abs(aq_value_EI) < self.EI_epsilon:
            print(f"!!! Exiting early. Evaluated {self.meta_ds_train_indices.shape[0]} programs.")
            return None


        self.last_suggestion_prog_idx = new_suggestion_prog_idx
        self.last_suggestion_idx_in_meta_ds_test = new_suggestion_idx_in_meta_ds_test
        self.last_suggestion_partial_prog_strs = new_suggestion_program_strs
        return new_suggestion_program_strs, aq_value_UCB

    def evaluate_suggested_program(self, eval_function: Callable[[str], EvaluatePathResult]) -> EvaluatePathResult:
        # evaluate the last suggestion
        assert self.last_suggestion_prog_idx is not None
        assert self.last_suggestion_idx_in_meta_ds_test is not None
        assert self.last_suggestion_partial_prog_strs is not None

        suggested_partial_program_strs = self.last_suggestion_partial_prog_strs
        total_num_module_layers = self.problem.architecture_class.get_num_modules()
        suggested_program_strs = (None,) * (total_num_module_layers - len(suggested_partial_program_strs)) \
                                 + suggested_partial_program_strs

        c_eval_result = eval_function(suggested_program_strs)

        # update the meta training datasets

        # remove from test dataset
        self.meta_ds_test_indices = np.delete(self.meta_ds_test_indices,
                                              self.last_suggestion_idx_in_meta_ds_test, axis=0)

        self.meta_ds_train_indices = np.vstack([self.meta_ds_train_indices, [self.last_suggestion_prog_idx]])
        self.meta_ds_train_out = np.vstack([self.meta_ds_train_out, [c_eval_result.val_loss]])
        # c_eval_result.val_loss
        # update the indices
        idx_of_best_in_meta_ds = np.argmin(self.meta_ds_train_out, axis=0)
        self.index_of_best_path = self.meta_ds_train_indices[idx_of_best_in_meta_ds].item()

        self.last_suggestion_prog_idx = None
        self.last_suggestion_idx_in_meta_ds_test = None
        self.last_suggestion_partial_prog_strs = None

        return c_eval_result


class PICLE_NTSearch:
    def __init__(self,
                 problem: Problem,
                 lib: PICLELibrary,
                 standalone_prog_trained_modules: List[NNModule],
                 standalone_prog_performance: float,
                 device: str,
                 search_for_full_programs_as_well: bool = False,
                 l_min = 3,
                 dbg_top_down_search_lmin_only=False
                 ):
        """
        :param l_min: minimum number of modules that need to be transferred to have a impactful knowledge transfer.
        """
        self.problem = problem
        self.lib = lib
        self.standalone_prog_trained_modules = standalone_prog_trained_modules
        self.standalone_prog_performance = standalone_prog_performance
        self.device = device
        self.search_for_full_programs_as_well = search_for_full_programs_as_well

        self.architecture_class = problem.architecture_class
        self.num_module_layers = self.architecture_class.get_num_modules()
        self.l_min = l_min

        self.dbg_top_down_search_lmin_only = dbg_top_down_search_lmin_only

        c_architecture_module_types = self.architecture_class.get_module_types()
        self.past_transferrable_solutions = []
        self.past_transferrable_solutions_only_last_layers = []

        for c_past_optimal_path in self.lib.previous_solutions:
            c_past_optimal_path_module_types = [self.lib[cpop_cmn].module_type for cpop_cmn in c_past_optimal_path]

            smaller_path_len = min(len(c_architecture_module_types), len(c_past_optimal_path_module_types))

            c_path_overlapping_module_types_module_names = []
            # from the last layer to first, see how many match.
            for c_module_layer_idx_in_reverse in range(1, smaller_path_len+1):
                if c_architecture_module_types[-c_module_layer_idx_in_reverse] == c_past_optimal_path_module_types[-c_module_layer_idx_in_reverse]:
                    c_module_name = c_past_optimal_path[-c_module_layer_idx_in_reverse]
                    c_path_overlapping_module_types_module_names.insert(0, c_module_name)
                else:
                    break

            # if none overlap, or if the number overlapping is smaller than l_min, skip.
            if len(c_path_overlapping_module_types_module_names) == 0 or \
                    len(c_path_overlapping_module_types_module_names) < l_min:
                continue

            c_transferrable_path = tuple(c_path_overlapping_module_types_module_names)
            c_transferrable_path_only_last_layers = c_transferrable_path[-l_min:]

            if c_transferrable_path_only_last_layers in self.past_transferrable_solutions_only_last_layers:
                # If there are duplicates, only use the first one
                continue

            self.past_transferrable_solutions.append(c_transferrable_path)
            self.past_transferrable_solutions_only_last_layers.append(c_transferrable_path_only_last_layers)

        if len(self.past_transferrable_solutions_only_last_layers) == 0:
            # if there is nothing to transfer, skip everything, don't recommend anything.
            self.past_solutions_selector = None
            self.best_past_solution_idx_selected = True
            self.past_solution_selected_paths_left_to_explore = []
        else:
            self.past_solutions_selector = PICLE_NTPathSelector_BOpt(problem, lib,
                                                                     self.past_transferrable_solutions_only_last_layers,
                                                                     standalone_prog_trained_modules, standalone_prog_performance,
                                                                     device)
            self.best_past_solution_idx_selected = False
            self.past_solution_selected_paths_left_to_explore = None

    def suggest_next_program(self):
        if self.best_past_solution_idx_selected and len(self.past_solution_selected_paths_left_to_explore) == 0:
            return None

        if not self.best_past_solution_idx_selected:
            c_suggestion = self.past_solutions_selector.suggest_next_program()

            if c_suggestion is None:
                self.best_past_solution_idx_selected = True
                self.past_solution_selected_paths_left_to_explore = []

                if self.dbg_top_down_search_lmin_only:
                    return None

                if self.past_solutions_selector.index_of_best_path == len(self.past_transferrable_solutions):
                    # the standalone is best. no point going further.
                    return None

                best_past_solution = self.past_transferrable_solutions[self.past_solutions_selector.index_of_best_path]
                # generate the possible different sub-paths to transfer:
                for last_l_to_transfer in range(self.l_min+1, len(best_past_solution)+1):
                    c_upper_part = best_past_solution[-last_l_to_transfer:]
                    complete_path = (None,) * (self.num_module_layers - len(c_upper_part)) + c_upper_part
                    self.past_solution_selected_paths_left_to_explore.append(complete_path)

                if len(self.past_solution_selected_paths_left_to_explore) == 0:
                    return None
            else:
                return c_suggestion

        return self.past_solution_selected_paths_left_to_explore[0]

    def evaluate_suggested_program(self, eval_function: Callable[[str], EvaluatePathResult]):
        if self.best_past_solution_idx_selected:
            last_suggestion_program_strs = self.past_solution_selected_paths_left_to_explore.pop(0)
            return eval_function(last_suggestion_program_strs)
        else:
            return self.past_solutions_selector.evaluate_suggested_program(eval_function)

