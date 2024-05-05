from torch.nn import Module
from typing import List, Tuple, Type, Union
from abc import ABC, abstractmethod
import numpy as np


class NNModule(Module):
    """
    An abstract base class, specifying the interface of a NN module.
    The logging methods allow for the inputs and outputs of a module to be recorded
    during forward propagation.
    """
    def __init__(self):
        super().__init__()
        self._logged_inputs_np_list = []
        self._logged_outputs_np_list = []
        self._log_io = False

    def start_logging_io(self):
        self._log_io = True
        self._logged_inputs_np_list = []
        self._logged_outputs_np_list = []

    def stop_logging_io_and_clear_logs(self):
        self._log_io = False
        self._logged_inputs_np_list = []
        self._logged_outputs_np_list = []

    def get_logged_io(self) -> Tuple[np.ndarray, np.ndarray]:
        logged_i = np.vstack(self._logged_inputs_np_list)
        logged_o = np.vstack(self._logged_outputs_np_list)

        return logged_i, logged_o

    def forward(self, x, return_logits_as_well=False):
        if return_logits_as_well:
            l, o = self._forward(x, return_logits_as_well=True)
        else:
            l = None
            o = self._forward(x, return_logits_as_well=False)

        if self._log_io:
            x_np = x.cpu().detach().numpy()
            o_np = o.cpu().detach().numpy()

            self._logged_inputs_np_list.append(x_np)
            self._logged_outputs_np_list.append(o_np)

        if return_logits_as_well:
            return l, o
        else:
            return o

    @abstractmethod
    def _forward(self, x, return_logits_as_well=False):
        pass

    @staticmethod
    @abstractmethod
    def get_module_type_name():
        pass

    @staticmethod
    def needs_output_dim():
        return False
