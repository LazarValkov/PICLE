from abc import ABC, abstractmethod
from typing import List, Tuple, Type, Union
import torch
from torch.utils.data import DataLoader
import numpy as np
from CL.Interface.Hypothesis import Hypothesis
from CL.Interface.NNModule import NNModule


class NNArchitecture(Hypothesis):
    """
    Defines a NN Architecture, specifying the types of modules involved and other helper functions.
    Can be instantiated to represent a NN with this architecture.
    """
    @staticmethod
    @abstractmethod
    def get_criterion_class():
        """
        :return: The class of the loss function which is to be used during training
        """
        pass

    @staticmethod
    @abstractmethod
    def get_num_modules() -> int:
        pass

    @staticmethod
    @abstractmethod
    def get_module_types() -> List[Type[NNModule]]:
        """
        :return: a list of the module classes which this architecture combines
        """
        pass

    # @staticmethod
    @abstractmethod
    def create_new_module_by_index(self, index) -> NNModule:
        pass

    @staticmethod
    @abstractmethod
    def get_partial_program_outputs_PT(partial_program: Tuple[NNModule], data_loader: DataLoader, device: str) -> np.ndarray:
        """
        Calculates the outputs of a partial program, using the given inputs
        :param partial_program:
        :param data_loader: a DataLoader which provides the inputs for the partial program
        :param device: the device to which the data should be ported
        :return: a numpy array, containing samples the output distribution of the partial program, computed using the given training dataset
        """
        pass

    @staticmethod
    @abstractmethod
    def get_partial_program_outputs_HLT(partial_program: Tuple[NNModule], data_loader: DataLoader, device: str,
                                        use_logits=False) -> np.ndarray:
        # calculate the outputs of a partial program using the given inputs
        pass

    @staticmethod
    @abstractmethod
    def get_layers_input_shape(pp_first_module_layer_idx: int):
        pass

    def __init__(self, modules: Tuple[Union[NNModule, None], ...], device: str, enable_finetuning: bool,
                 if_finetuning_finetune_a_copy: bool, random_init_random_seed: Union[int, None],
                 output_dim=None):
        self.enable_finetuning = enable_finetuning
        self.if_finetuning_finetune_a_copy = if_finetuning_finetune_a_copy
        self.device = device
        self.output_dim = output_dim

        # needed to ensure the same random weight initialization
        if random_init_random_seed is not None:
            last_state = torch.get_rng_state()
            torch.manual_seed(random_init_random_seed)

        self.modules = []
        self.trainable_modules: List[NNModule] = []  # not necessarily ordered by when they are used
        self.new_modules: List[NNModule] = []

        for i in range(self.get_num_modules()):
            c_module = modules[i]

            if c_module is None:
                c_module = self.create_new_module_by_index(i) # .to(device)
                c_module = c_module.to(device)
                self.trainable_modules.append(c_module)
                self.new_modules.append(c_module)
                self.modules.append(c_module)
            else:
                if self.enable_finetuning:
                    if self.if_finetuning_finetune_a_copy:
                        # create a copy of the module
                        c_module_cpy = self.create_new_module_by_index(i).to(device)
                        c_module_cpy.load_state_dict(c_module.state_dict())
                        c_module = c_module_cpy
                        self.new_modules.append(c_module)
                    self.trainable_modules.append(c_module)
                self.modules.append(c_module)

        if random_init_random_seed is not None:
            torch.set_rng_state(last_state)

    def get_trainable_parameters(self):
        params_to_optimise = []
        for m in self.trainable_modules:
            params_to_optimise += list(m.parameters())
        return params_to_optimise

    def has_trainable_parameters(self):
        return len(self.trainable_modules) > 0

    def get_trainable_parameters_apart_last_module(self):
        params_to_optimise = []
        for m in self.trainable_modules[:-1]:
            params_to_optimise += list(m.parameters())
        return params_to_optimise

    def get_trainable_named_parameters_apart_last_module(self):
        all_tuples = []

        for m_idx, m in enumerate(self.trainable_modules[:-1]):
            c_module_name = m.get_module_type_name()
            for name, param in m.named_parameters(str(m_idx) + "_"):
                name_updated = c_module_name + "_" + name
                all_tuples.append((name_updated, param))

        return all_tuples

    def train(self):
        for m in self.trainable_modules:
            m.train()

    def eval(self):
        for m in self.modules:
            m.eval()

    def __call__(self, inputs):
        return self.forward(inputs)

    @abstractmethod
    def forward(self, inputs):
        pass

    def save_to_a_single_file(self, filepath: str):
        all_state_dicts = [m.state_dict() for m in self.modules]
        torch.save(all_state_dicts, filepath)

    def load_modules_from_a_single_file(self, filepath: str, device: str):
        all_state_dicts = torch.load(filepath, map_location=torch.device(device))
        assert len(all_state_dicts) == len(self.modules)
        for i, state_dict in enumerate(all_state_dicts):
            self.modules[i].load_state_dict(state_dict)