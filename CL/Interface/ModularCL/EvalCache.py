import os
import pickle
from pathlib import Path
from typing import Union, Tuple, List, Type
import torch
from CL.Interface.NNArchitecture import NNArchitecture


class EvalCache:
    def __init__(self, folder: Union[None, str], _debug_eval_cache_throw_exception_if_cant_load_module):
        self.folder = folder
        self._debug_eval_cache_throw_exception_if_cant_load_module = _debug_eval_cache_throw_exception_if_cant_load_module

        # create the folder if it doesn't exist
        Path(folder).mkdir(parents=True, exist_ok=True)

        self.c_eval_dict_filepath = f"{folder}/eval_dict.pkl"

        if os.path.exists(self.c_eval_dict_filepath):
            with open(self.c_eval_dict_filepath, 'rb') as fh:
                self.eval_cache_dict = pickle.load(fh)
        else:
            self.eval_cache_dict = {}

    def keys(self):
        return self.eval_cache_dict.keys()

    # c_module_name = f"{c_module.get_module_type_name()}_{problem.name}"
    def add_evaluation(self, c_program_str: str,
                       exploration_time: float, val_loss: float, val_acc: float,
                       new_modules, learning_curves_dict):
        self.eval_cache_dict[c_program_str] = {
            'exploration_time': exploration_time,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_curves_dict': learning_curves_dict
        }

        # save the new modules, so we don't have to keep them in memory
        for module in new_modules:
            c_filename = f"{c_program_str}_{module.get_module_type_name()}.pth"
            c_filename = c_filename.replace(" ", "_")

            c_filepath = f"{self.folder}/{c_filename}"
            torch.save(module.state_dict(), c_filepath)

    def load_newly_trained_modules(self, c_program_str: str, model: NNArchitecture):
        assert c_program_str in self.eval_cache_dict.keys()

        for c_module in model.trainable_modules:
            c_filename = f"{c_program_str}_{c_module.get_module_type_name()}.pth"
            c_filename = c_filename.replace(" ", "_")

            c_filepath = f"{self.folder}/{c_filename}"

            if os.path.exists(c_filepath):
                c_module.load_state_dict(
                    torch.load(c_filepath, map_location=torch.device(model.device)))
            elif self._debug_eval_cache_throw_exception_if_cant_load_module:
                raise ValueError()
            else:
                pass

    def __getitem__(self, program_str: str):
        return self.eval_cache_dict[program_str]

    def save(self):
        with open(self.c_eval_dict_filepath, 'wb') as f:
            pickle.dump(self.eval_cache_dict, f, pickle.HIGHEST_PROTOCOL)
