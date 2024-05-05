from random import Random
import math
import time
from typing import Union, Dict, Tuple, List
import numpy as np
import torch

from CL.Interface.NNArchitecture import NNArchitecture
from CL.Interface.Problem import Problem
from CL.Interface.NNTrainer import NNTrainer
from CL.Interface.NNEvaluator import NNEvaluator
from CL.PICLE.Library import PICLELibrary
from CL.Interface.ModularCL.EvalCache import EvalCache

from CL.Interface.ModularCL.ModularCLAlgorithm import ModularCLAlg
from CL.PICLE.PTSearch import PICLE_PTSearch
from CL.PICLE.NTSearch import PICLE_NTSearch


class PICLE(ModularCLAlg):
    def __init__(self,
                 pt_hyperparam_soft_type_target_dim: int,
                 pt_hyperparam_softmax_temp: float,
                 nt_hyperparam_num_input_points_stored_per_module: int,
                 nt_hyperparam_lmin: int,
                 *args,
                 _debug_search_perceptual_transfer=True,
                 _debug_search_latent_transfer=True,
                 _debug_verbose: bool = False,
                 _debug_top_down_search_lmin_only=False,
                 **kwargs):
        """
        :param pt_hyperparam_soft_type_target_dim: the dimensionality which the inputs are randomly projected to, before estimating their distribution
        :param pt_hyperparam_softmax_temp: The softmax temperature used for the prior over modules, for PT search
        :param nt_hyperparam_num_input_points_stored_per_module: num inputs per module stored for the NT search to compute funcitonal distances. Can be further optimized and only saved for the modules at the L-l_min+1 layer.
        :param nt_hyperparam_lmin: l_min used for NT search
        :param _debug_search_perceptual_transfer: if True, enables PT search
        :param _debug_search_latent_transfer: if True, enables NT search
        :param _debug_verbose: if True, prints (more) stuff
        :param _debug_top_down_search_lmin_only: if True, only transfer l_min modules for NT transfer (not more)
        """
        # A bunch of assertions to make sure I don't get the hyperparameters wrong in the future
        self._debug_verbose = _debug_verbose
        self.soft_type_target_dim = pt_hyperparam_soft_type_target_dim
        self.num_input_points_stored_per_module = nt_hyperparam_num_input_points_stored_per_module
        self.pt_hyperparam_softmax_temp = pt_hyperparam_softmax_temp
        self.nt_hyperparam_lmin = nt_hyperparam_lmin
        self._debug_top_down_search_lmin_only = _debug_top_down_search_lmin_only
        self._debug_search_perceptual_transfer = _debug_search_perceptual_transfer
        self._debug_search_latent_transfer = _debug_search_latent_transfer

        assert self._debug_search_perceptual_transfer or self._debug_search_latent_transfer
        super().__init__(*args, **kwargs)

    def get_new_library(self, device) -> PICLELibrary:
        return PICLELibrary(device, self.soft_type_target_dim, self.num_input_points_stored_per_module)

    def get_name(self):
        if self._debug_search_perceptual_transfer and not self._debug_search_latent_transfer:
            return f"PICLE_PT"

        if not self._debug_search_perceptual_transfer and self._debug_search_latent_transfer:
            return "PICLE_NT"

        return "PICLE"

    def iterate_programs(self, problem: Problem, trainer: NNTrainer, evaluator: NNEvaluator,
                         eval_cache: Union[EvalCache, None],
                         random_obj: Random, random_init_random_seed: int):
        eval_function = lambda modules_str: self.explore_program(modules_str, problem, trainer, evaluator,
                                                                 eval_cache, random_init_random_seed)
        # evaluate the standalone version
        c_num_module_layers = problem.architecture_class.get_num_modules()
        sa_program = (None,) * c_num_module_layers
        if self._debug_verbose:
            print(f"evaluating program={sa_program}")

        standalone_eval_res = eval_function(sa_program)
        yield standalone_eval_res

        # initiate this every time, since this is a problem-specific object (the way it's written)
        if self._debug_search_perceptual_transfer:
            picle_pt_searcher = PICLE_PTSearch(problem, self.lib, self.soft_type_target_dim,
                                               self.device,
                                               standalone_eval_res.val_loss,
                                               self._debug_verbose,
                                               prior_softmax_temp=self.pt_hyperparam_softmax_temp)
            # Evaluate PT paths suggestions
            c_suggestion_time_start = time.time()
            suggestion_pt = picle_pt_searcher.suggest_next_program()
            while suggestion_pt is not None:
                if self._debug_include_suggestion_time:
                    suggestion_time = time.time() - c_suggestion_time_start
                else:
                    suggestion_time = 0.
                c_res = picle_pt_searcher.evaluate_suggested_program(eval_function)
                c_res.exploration_time += suggestion_time
                yield c_res

                c_suggestion_time_start = time.time()
                suggestion_pt = picle_pt_searcher.suggest_next_program()

        if self._debug_search_latent_transfer:
            picle_nt_searcher = PICLE_NTSearch(problem,
                                               self.lib,
                                               standalone_eval_res.model.trainable_modules,
                                               standalone_eval_res.val_loss,
                                               self.device,
                                               search_for_full_programs_as_well=True,
                                               l_min=self.nt_hyperparam_lmin,
                                               dbg_top_down_search_lmin_only=self._debug_top_down_search_lmin_only)
            # Evaluate NT paths suggestions
            c_suggestion_time_start = time.time()
            suggestion_nt = picle_nt_searcher.suggest_next_program()
            while suggestion_nt is not None:
                if self._debug_include_suggestion_time:
                    suggestion_time = time.time() - c_suggestion_time_start
                else:
                    suggestion_time = 0.

                c_res = picle_nt_searcher.evaluate_suggested_program(eval_function)
                c_res.exploration_time += suggestion_time
                yield c_res

                c_suggestion_time_start = time.time()
                suggestion_nt = picle_nt_searcher.suggest_next_program()

    def update_lib(self, problem: Problem, final_model: NNArchitecture, performance):
        # add the most successful modules to the library
        final_model.eval()
        modules_to_log = final_model.trainable_modules
        for m in modules_to_log:
            m.start_logging_io()
        # go through the whole training dataset to log the inputs and outputs of the new modules
        with torch.no_grad():
            for data, _ in problem.get_tr_data_loader():
                data = data.to(self.device)
                final_model(data)

        for c_module in modules_to_log:
            c_module_type = type(c_module)
            if c_module_type not in self.lib.items_by_module_type.keys():
                matching_modules_in_lib = []
            else:
                matching_modules_in_lib = [li.name for li in self.lib.items_by_module_type[type(c_module)] if c_module == li.module]  # self.lib  test_list.count(15)
            assert len(matching_modules_in_lib) in [0, 1]
            module_already_in_library = len(matching_modules_in_lib) == 1

            # if the module is not in the library, add it to the library
            if not module_already_in_library:
                c_module_name = f"{problem.num_id}_{c_module.get_module_type_name()}"
                c_module_logged_inputs, c_module_logged_outputs = c_module.get_logged_io()
                self.lib.add_module(c_module_name, c_module, c_module_logged_inputs, c_module_logged_outputs, performance)
            else:
                # otherwise, add its new soft type only.
                c_module_name = matching_modules_in_lib[0]
                c_module_logged_inputs, c_module_logged_outputs = c_module.get_logged_io()
                self.lib.add_soft_type_to_module(c_module_name, c_module_logged_inputs, c_module_logged_outputs, performance)

        for m in modules_to_log:
            m.stop_logging_io_and_clear_logs()

