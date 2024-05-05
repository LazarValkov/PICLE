from typing import Tuple, Union, List, Optional
import time
import itertools
from random import Random

from CL.Interface.ModularCL.EvalCache import EvalCache
from CL.Interface.ModularCL.ModularCLAlgorithm import ModularCLAlg
from CL.Interface.Problem import Problem
from CL.Interface.NNEvaluator import NNEvaluator
from CL.Interface.NNTrainer import NNTrainer
from CL.Interface.ModularCL.Library import Library


class HOUDINI(ModularCLAlg):
    # Since we assume the architecture is given, HOUDINI is equal to exhaustive search
    def get_name(self):
        return "HOUDINI"

    def get_should_limit_evaluations_by_number_of_paths(self):
        return True

    def get_new_library(self, device) -> Library:
        return Library(device)

    def _get_modules_names_list_of_lists(self, problem: Problem) -> List[List[str]]:
        """
        retrieves the relevant module names from the library and organises them into a list of lists,
        grouping them by module types.
        """
        modules_types_list = problem.architecture_class.get_module_types()
        modules_names_list_of_lists = []
        for c_module_type in modules_types_list:
            if c_module_type not in self.lib.items_by_module_type.keys():
                # there are not library items of this type.
                c_module_names_list = []
            else:
                c_module_names_list = [lib_item.name for lib_item in self.lib.items_by_module_type[c_module_type]]
            modules_names_list_of_lists.append(c_module_names_list)
        return modules_names_list_of_lists

    @staticmethod
    def exh_search_iteration_bottom_up(items_lists, LLT=False):
        if len(items_lists) == 0:
            return [tuple()]

        return_list_tuples = []
        c_item_list = items_lists[0]
        for item in c_item_list:
            if LLT and item is None:
                c_tuple = (None,) * len(items_lists)
                return_list_tuples.append(c_tuple)
                continue

            for following_items_tuple in HOUDINI.exh_search_iteration_bottom_up(items_lists[1:], LLT=LLT):
                c_tuple = (item,) + following_items_tuple
                return_list_tuples.append(c_tuple)
        return return_list_tuples

    def _get_all_module_combinations(self, problem: Problem, random_obj: Random):
        modules_names_list_of_lists = self._get_modules_names_list_of_lists(problem)
        for mnl in modules_names_list_of_lists:
            mnl.append(None)
        all_module_combinations = self.exh_search_iteration_bottom_up(modules_names_list_of_lists)

        standalone_prog = (None,) * problem.architecture_class.get_num_modules()
        all_module_combinations.remove(standalone_prog)
        all_module_combinations.insert(0, standalone_prog)

        return all_module_combinations

    def iterate_programs(self, problem: Problem, trainer: NNTrainer, evaluator: NNEvaluator,
                         eval_cache: Union[None, EvalCache],
                         random_obj: Random, random_init_random_seed):
        """
        yield program_str, loss, acc, exploration_time, newly_trained_modules
        """
        c_suggestion_time_start = time.time()

        all_module_combinations = self._get_all_module_combinations(problem, random_obj)

        for p in all_module_combinations:
            if self._debug_include_suggestion_time:
                suggestion_time = time.time() - c_suggestion_time_start
            else:
                suggestion_time = 0.

            c_prog_exploration_res = self.explore_program(p, problem, trainer, evaluator, eval_cache, random_init_random_seed)
            c_prog_exploration_res.exploration_time += suggestion_time
            yield c_prog_exploration_res

            c_suggestion_time_start = time.time()
