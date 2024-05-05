from typing import Optional, Dict, Tuple
from random import Random
import operator
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import copy
from functools import reduce

from CL.Baselines.HOUDINI import HOUDINI
from CL.Interface.Problem import Problem


def flatten(x):
    n_features = reduce(operator.mul, x.size()[1:])
    return x.reshape(x.shape[0], n_features)


class MNTDP(HOUDINI):
    def __init__(self, *args, k=1, **kwargs):
        super(MNTDP, self).__init__(*args, **kwargs)
        # this is updated by _get_all_module_combinations
        self.program_str_to_program_modules_strs_in_updated_lib: Dict[str, Tuple[str, ...]] = {}

    def get_should_limit_evaluations_by_number_of_paths(self):
        return False

    def get_name(self):
        return f"MNTDP"

    def _get_acc_of_solution(self, problem: Problem, c_prog_strs):
        # get the program
        c_modules = tuple(self.lib[s].module for s in c_prog_strs)
        num_modules = problem.architecture_class.get_num_modules()
        if len(c_modules) < num_modules:
            # if this program is shorter than expected, fill the rest of the modules with randomly initialised modules
            additional_modules = [None,] * (num_modules-len(c_modules))
            c_modules = c_modules + tuple(additional_modules)
            c_model = problem.architecture_class(c_modules, self.device,
                                                  enable_finetuning=False, if_finetuning_finetune_a_copy=True,
                                                  random_init_random_seed=777,
                                              output_dim=problem.output_dim)
            c_modules_except_last = c_model.modules[:-1]
        else:
            assert len(c_modules) == num_modules
            c_modules_except_last = c_modules[:-1]

        # c_modules_except_last = c_modules[:-1]

        # get the features on the training dataset
        tr_features, tr_targets = [], []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(problem.get_tr_data_loader()):
                data = data.to(self.device)
                c_feat = problem.architecture_class._forward_pp_PT(c_modules_except_last, data)

                c_feat_np = c_feat.detach().cpu().numpy()
                if len(c_feat_np.shape) > 2:
                    c_feat_np = c_feat_np.reshape((c_feat_np.shape[0], -1))

                tr_features.append(c_feat_np)
                tr_targets.append(target.numpy().reshape((-1, 1)))

        tr_features = np.vstack(tr_features)
        tr_targets = np.vstack(tr_targets).reshape((-1,))

        # fit a KNN model
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(tr_features, tr_targets)

        # evaluate the accuracy on the val dataset
        val_knn_predictions, val_targets = [], []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(problem.get_val_data_loader()):
                data = data.to(self.device)
                c_feat = problem.architecture_class._forward_pp_PT(c_modules_except_last, data)

                c_feat_np = c_feat.detach().cpu().numpy()
                if len(c_feat_np.shape) > 2:
                    c_feat_np = c_feat_np.reshape((c_feat_np.shape[0], -1))

                knn_preds = knn.predict(c_feat_np)

                val_knn_predictions.append(knn_preds)
                val_targets.append(target.numpy().reshape((-1, 1)))

        val_knn_predictions = np.hstack(val_knn_predictions).reshape((-1,))
        val_targets = np.vstack(val_targets).reshape((-1,))
        knn_val_acc = (val_knn_predictions == val_targets).astype(np.float32).mean().item()
        return knn_val_acc

    def _get_closest_solution(self, problem: Problem, solutions_to_all_previous_problems):
        c_solutions = copy.copy(solutions_to_all_previous_problems)
        sol_to_acc = {
            s: self._get_acc_of_solution(problem, s) for s in c_solutions
        }
        best_sol = max(sol_to_acc, key=sol_to_acc.get)

        return best_sol

    def _get_all_module_combinations(self, problem: Problem, random_obj: Random):
        solutions_to_all_previous_problems = self.lib.previous_solutions

        # 1) choose the closest previous problem-solution using KNN
        if len(solutions_to_all_previous_problems) > 0:
            closest_solution = self._get_closest_solution(problem, solutions_to_all_previous_problems)
        else:
            closest_solution = tuple()

        # reset this dictionary so that it's problem-specific
        self.program_str_to_program_modules_strs_in_updated_lib = {}

        # 2) Add the possible module combinations using this and None (only PT)
        num_modules_in_prog = problem.architecture_class.get_num_modules()
        all_module_combinations = []

        standalone_prog = (None,) * num_modules_in_prog
        standalone_prog_str = ' '.join((str(j) for j in standalone_prog))
        standalone_prog_strs_in_updated_library = tuple(
            f"{problem.num_id}_{problem.architecture_class.get_module_types()[i].get_module_type_name()}"
            for i in range(num_modules_in_prog))


        self.program_str_to_program_modules_strs_in_updated_lib[
            standalone_prog_str] = standalone_prog_strs_in_updated_library

        # Evaluate standalone first, for more comparable results
        all_module_combinations.append(standalone_prog)

        for i in range(1, len(closest_solution)+1):
            # transfer the first i layers
            c_transfer_prog = closest_solution[:i] + (None,)*(num_modules_in_prog - i)
            c_transfer_prog_str = ' '.join((str(j) for j in c_transfer_prog))
            c_transfer_prog_strs_in_updated_lib = closest_solution[:i] + standalone_prog_strs_in_updated_library[i:]

            all_module_combinations.append(c_transfer_prog)
            self.program_str_to_program_modules_strs_in_updated_lib[
                c_transfer_prog_str] = c_transfer_prog_strs_in_updated_lib

        return all_module_combinations

