from abc import abstractmethod
import numpy as np
from typing import List, Tuple, Type, Union
import torch
from torch.utils.data import DataLoader
from CL.Interface.NNArchitecture import NNArchitecture
from Experiments.BELL.NNModules import *


class T1Architecture1(NNArchitecture):
    """
    An Architecture, which only uses CNN()
    """
    NUM_MODULES = 5

    @staticmethod
    def get_criterion_class():
        return torch.nn.CrossEntropyLoss

    @staticmethod
    def get_num_modules() -> int:
        return T1Architecture1.NUM_MODULES

    @staticmethod
    @abstractmethod
    def get_last_modules_type() -> Type[NNModule]:
        return T1Module_CNN_FCL3

    @classmethod
    def get_module_types(cls) -> List[Type[NNModule]]:
        return [T1Module_CNN_ConvL1, T1Module_CNN_ConvL2, T1Module_CNN_FCL1, T1Module_CNN_FCL2,
                cls.get_last_modules_type()]

    # @classmethod
    def create_new_module_by_index(self, index) -> NNModule:
        return self.get_module_types()[index]()

    @staticmethod
    def _forward_partial_program(partial_program: Tuple[NNModule],
                                 starting_layer_index,
                                 x: torch.Tensor, return_logits_as_well=False):
        assert 1 <= len(partial_program) <= T1Architecture1.NUM_MODULES

        c_logits = c_outputs = x
        for c_pp_layer_idx, c_module in enumerate(partial_program):
            c_layer_idx = starting_layer_index + c_pp_layer_idx
            if c_layer_idx == 2 and starting_layer_index != 2:
                c_outputs = torch.flatten(c_outputs, 1)
            c_logits, c_outputs = c_module(c_outputs, return_logits_as_well=True)

        ret_value = (c_logits, c_outputs) if return_logits_as_well else c_outputs
        return ret_value

    @staticmethod
    def get_partial_program_outputs_PT(partial_program: Tuple[NNModule],
                                       data_loader: DataLoader,
                                       device: str) -> np.ndarray:
        assert 1 <= len(partial_program) <= T1Architecture1.NUM_MODULES

        all_predictions = []
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            with torch.no_grad():
                c_pred_outs = T1Architecture1._forward_partial_program(partial_program, 0,
                                                                       data,
                                                                       return_logits_as_well=False)
            all_predictions.append(c_pred_outs.cpu().detach().numpy())

        all_predictions = np.vstack(all_predictions)
        return all_predictions

    @staticmethod
    def _forward_pp_PT(partial_program, c_inputs):
        return T1Architecture1._forward_partial_program(partial_program, 0, c_inputs, return_logits_as_well=False)

    @staticmethod
    def get_partial_program_outputs_HLT(partial_program: Tuple[NNModule],
                                        data_loader: DataLoader,
                                        device: str, use_logits=False) -> np.ndarray:
        assert 1 <= len(partial_program) <= T1Architecture1.NUM_MODULES
        starting_index = T1Architecture1.NUM_MODULES - len(partial_program)

        all_predictions = []
        for batch_idx, (data,) in enumerate(data_loader):
            data = data.to(device)
            with torch.no_grad():
                c_pred_logits, c_pred_outs = T1Architecture1._forward_partial_program(partial_program,
                                                                                      starting_index,
                                                                                      data,
                                                                                      return_logits_as_well=True)
            to_add = c_pred_logits if use_logits else c_pred_outs
            to_add = to_add.cpu().detach().numpy()
            all_predictions.append(to_add)

        all_predictions = np.vstack(all_predictions)
        return all_predictions

    @staticmethod
    def get_partial_program_normalized_input_gradients_HLT(partial_program: Tuple[NNModule],
                                                           data_loader: DataLoader,
                                                           device: str) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def get_layers_input_shape(pp_first_module_layer_idx: int):
        return {
            0: (1, 28, 28),
            1: (64, 12, 12),
            # (64, 4, 4) -> The input to torch.flatten()
            2: (1024,),
            3: (64,),
            4: (64,)
        }[pp_first_module_layer_idx]

    def __init__(self, modules: Tuple[Union[NNModule, None]], device: str,
                 enable_finetuning: bool, if_finetuning_finetune_a_copy: bool,
                 random_init_random_seed: Union[int, None],
                 output_dim=None):
        assert len(modules) == self.NUM_MODULES
        super().__init__(modules, device, enable_finetuning, if_finetuning_finetune_a_copy, random_init_random_seed,
                         output_dim=output_dim)

    def forward(self, inputs):
        return self._forward_partial_program(self.modules, 0,
                                             inputs,
                                             return_logits_as_well=True)


class T1Architecture2(NNArchitecture):
    """
    An Architecture, which combines 1 CNN and 1 FC layers
    """
    NUM_MLP_MODULES = 3
    NUM_MODULES = T1Architecture1.NUM_MODULES + NUM_MLP_MODULES

    @staticmethod
    def get_criterion_class():
        return torch.nn.BCEWithLogitsLoss  # return F.binary_cross_entropy_with_logits

    @staticmethod
    def get_num_modules() -> int:
        return T1Architecture2.NUM_MODULES

    @staticmethod
    def get_module_types() -> List[Type[NNModule]]:
        return T1Architecture1.get_module_types() + [
            T1Module_MLP_FCL1, T1Module_MLP_FCL2, T1Module_MLP_FCL3
        ]

    # @staticmethod
    def create_new_module_by_index(self, index) -> NNModule:
        return T1Architecture2.get_module_types()[index]()

    @staticmethod
    def _forward_pp_PT(partial_program, c_inputs):
        # since it's PT, we assume that the inputs are the inputs to the problem

        # case1: partial program includes only CNN components
        # case2: partial program includes CNN + MLP components
        includes_mlp = len(partial_program) > T1Architecture1.NUM_MODULES

        # first, get the CNN outputs
        pp_cnn_portion = partial_program[:T1Architecture1.NUM_MODULES]
        if type(c_inputs) == list:
            c_inputs_list = c_inputs
        else:
            c_inputs_list = [torch.squeeze(j, dim=1) for j in torch.split(c_inputs, 1, dim=1)]
        c_cnn_fn = lambda x: T1Architecture1._forward_partial_program(pp_cnn_portion,
                                                                      0,
                                                                      x, return_logits_as_well=False)
        cnn_outputs_list = list(map(c_cnn_fn, c_inputs_list))
        if not includes_mlp:
            return cnn_outputs_list

        mlp_partial_program = partial_program[T1Architecture1.NUM_MODULES:]
        inputs_to_mlp = torch.cat(cnn_outputs_list, dim=1)

        c_output = inputs_to_mlp
        for mlp_module in mlp_partial_program:
            c_output = mlp_module(c_output)

        return c_output

    @staticmethod
    def get_partial_program_outputs_PT(partial_program: Tuple[NNModule],
                                       data_loader: DataLoader,
                                       device: str) -> Union[np.ndarray, List[np.ndarray]]:
        # def get_partial_program_outputs(partial_program: Tuple[NNModule], data_loader: DataLoader, device: str) -> np.ndarray:
        # each input from data_loader is assumed to be a list of images. We apply the CNN on each image.

        all_predictions = []
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                data = data.to(device)
                with torch.no_grad():
                    c_pred = T1Architecture2._forward_pp_PT(partial_program, data)
                    if type(c_pred) == list:
                        # a list of tensors
                        c_pred_np = [li.cpu().detach().numpy() for li in c_pred]
                    else:
                        c_pred_np = c_pred.cpu().detach().numpy()
                all_predictions.append(c_pred_np)

        if type(all_predictions[0]) == np.ndarray:
            return np.vstack(all_predictions)

        assert type(all_predictions[0]) == list and type(all_predictions[0][0]) == np.ndarray
        # merge all the sublists concurrently
        new_list = []
        for i in range(len(all_predictions[0])):
            new_list.append(np.vstack([ap[i] for ap in all_predictions]))
        return new_list

    @staticmethod
    def _forward_pp_HLT(partial_program, c_inputs):
        # 0, 1, 2, 3, 4, 5
        starting_layer_index = T1Architecture2.NUM_MODULES - len(partial_program)

        # if the partial program involves CNN layers
        if starting_layer_index < T1Architecture1.NUM_MODULES:
            if starting_layer_index == 0:
                if type(c_inputs) != list:
                    c_inputs = [torch.squeeze(j, dim=1) for j in torch.split(c_inputs, 1, dim=1)]

            num_following_cnn_modules = T1Architecture1.NUM_MODULES - starting_layer_index
            partial_program_cnn_modules = partial_program[:num_following_cnn_modules]

            c_cnn_fn = lambda x: T1Architecture1._forward_partial_program(partial_program_cnn_modules,
                                                                          starting_layer_index,
                                                                          x, return_logits_as_well=False)
            cnn_outputs_list = list(map(c_cnn_fn, c_inputs))

            # concatenate
            inputs_to_mlp = torch.cat(cnn_outputs_list, dim=1)
        else:
            inputs_to_mlp = c_inputs

        mlp_partial_program = partial_program[-T1Architecture2.NUM_MLP_MODULES:]
        c_logits, c_output = inputs_to_mlp, inputs_to_mlp
        for mlp_module in mlp_partial_program:
            c_logits, c_output = mlp_module(c_output, return_logits_as_well=True)
        return c_logits, c_output

    @staticmethod
    def get_partial_program_outputs_HLT(partial_program: Tuple[NNModule], data_loader: DataLoader, device: str,
                                        use_logits=False) -> np.ndarray:
        assert 1 <= len(partial_program) <= T1Architecture2.NUM_MODULES

        all_outputs = []
        with torch.no_grad():
            for batch_idx, (data,) in enumerate(data_loader):
                if type(data) == list:
                    data = [d.to(device) for d in data]
                else:
                    data = data.to(device)
                c_pred_logits, c_pred_outs =  T1Architecture2._forward_pp_HLT(partial_program, data)

                to_add = c_pred_logits if use_logits else c_pred_outs
                to_add = to_add.cpu().detach().numpy()
                all_outputs.append(to_add)
            all_outputs_np = np.vstack(all_outputs)

        return all_outputs_np

    @staticmethod
    def get_partial_program_normalized_input_gradients_HLT(partial_program: Tuple[NNModule], data_loader: DataLoader,
                                                           device: str,
                                                           use_logits=True) -> np.ndarray:

        assert 1 <= len(partial_program) <= T1Architecture2.NUM_MODULES

        in_dim = np.prod(data_loader.dataset.tensors[0].shape[1:]).item()
        all_input_gradients = []

        for batch_idx, (data,) in enumerate(data_loader):
            if type(data) == list:
                data = [d.to(device) for d in data]
                for d in data:
                    d.requires_grad = True
            else:
                data = data.to(device)
                data.requires_grad = True

            for m in partial_program:
                m.zero_grad()

            c_pred_logits, c_pred_outs = T1Architecture2._forward_pp_HLT(partial_program, data)
            c_pred_logits_summed = c_pred_logits.sum()
            c_pred_logits_summed.backward()

            with torch.no_grad():
                c_grad_flat = torch.reshape(data.grad, (-1, in_dim))
                normalized_grads = c_grad_flat / torch.norm(c_grad_flat, dim=1, keepdim=True)
            normalized_grads_np = normalized_grads.cpu().detach().numpy()
            all_input_gradients.append(normalized_grads_np)

        for m in partial_program:
            m.zero_grad()

        all_input_gradients = np.vstack(all_input_gradients)
        return all_input_gradients

    @staticmethod
    def forward_program(inputs, program: Tuple[NNModule], return_logits_as_well=True):
        # inputs is a list of images, represented as a tensor
        # first, split it back into a list
        if type(inputs) == list:
            inputs_list = inputs
        else:
            inputs_list = [torch.squeeze(j, dim=1) for j in torch.split(inputs, 1, dim=1)]
        cnn_fn = lambda x: T1Architecture1._forward_partial_program(program[:T1Architecture1.NUM_MODULES], 0, x, return_logits_as_well=False)
        h1_list = list(map(cnn_fn, inputs_list))
        h1_concat = torch.cat(h1_list, dim=1)

        c_logits, c_outputs = h1_concat, h1_concat
        for m in program[T1Architecture1.NUM_MODULES:]:
            c_logits, c_outputs = m(c_outputs, return_logits_as_well=True)
        return (c_logits, c_outputs) if return_logits_as_well else c_outputs

    @staticmethod
    def get_layers_input_shape(pp_first_module_layer_idx: int):
        # the first item in the list indicates the list's length
        return {
            0: [T1_MODULES_LIST_LENGTH, 1, 28, 28],
            1: [T1_MODULES_LIST_LENGTH, 64, 12, 12],
            2: [T1_MODULES_LIST_LENGTH, 1024, ],
            3: [T1_MODULES_LIST_LENGTH, 64, ],
            4: [T1_MODULES_LIST_LENGTH, 64, ],

            5: (T1_MODULES_FC_IN_DIM,),
            6: (T1_FC_NUM_HIDDEN_UNITS,),
            7: (T1_MODULES_LAST_HIDDEN_STATE_NUM_UNITS,)
        }[pp_first_module_layer_idx]

    def __init__(self, modules: Tuple[Union[NNModule, None]], device: str,
                 enable_finetuning: bool, if_finetuning_finetune_a_copy: bool,
                 random_init_random_seed: Union[int, None],
                 output_dim=None):
        assert len(modules) == self.NUM_MODULES
        super().__init__(modules, device, enable_finetuning, if_finetuning_finetune_a_copy, random_init_random_seed, output_dim=output_dim)

    def forward(self, inputs):
        logits, outputs = self.forward_program(inputs, self.modules, return_logits_as_well=True)
        return logits, outputs


class T2Architecture1(NNArchitecture):
    """
    An Architecture, which only uses CNN()
    """
    NUM_MODULES = 4

    @staticmethod
    def get_criterion_class():
        return torch.nn.CrossEntropyLoss

    @staticmethod
    def get_num_modules() -> int:
        return T2Architecture1.NUM_MODULES

    @staticmethod
    @abstractmethod
    def get_last_modules_type() -> Type[NNModule]:
        pass

    @classmethod
    def get_module_types(cls) -> List[Type[NNModule]]:
        return [T2Module_MLP_FCL1, T2Module_MLP_FCL2, T2Module_MLP_FCL3, T2Module_MLP_FCL4]

    def create_new_module_by_index(self, index) -> NNModule:
        return self.get_module_types()[index]()

    @staticmethod
    def _forward_partial_program(partial_program: Tuple[NNModule],
                                 starting_layer_index,
                                 x: torch.Tensor, return_logits_as_well=False):

        raise NotImplementedError()

    @staticmethod
    def get_partial_program_outputs_PT(partial_program: Tuple[NNModule],
                                       data_loader: DataLoader,
                                       device: str) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def _forward_pp_PT(partial_program, c_inputs):
        # return T1Architecture1._forward_partial_program(partial_program, 0, c_inputs, return_logits_as_well=False)
        raise NotImplementedError()

    @staticmethod
    def get_partial_program_outputs_HLT(partial_program: Tuple[NNModule],
                                        data_loader: DataLoader,
                                        device: str, use_logits=False) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def get_partial_program_normalized_input_gradients_HLT(partial_program: Tuple[NNModule],
                                                           data_loader: DataLoader,
                                                           device: str) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def get_layers_input_shape(pp_first_module_layer_idx: int):
        return {
            0: (1, 28, 28),
            1: (64, 12, 12),
            # (64, 4, 4) -> The input to torch.flatten()
            2: (1024,),
            3: (64,),
            4: (64,)
        }[pp_first_module_layer_idx]

    def __init__(self, modules: Tuple[Union[NNModule, None]], device: str,
                 enable_finetuning: bool, if_finetuning_finetune_a_copy: bool,
                 random_init_random_seed: Union[int, None],
                 output_dim=None):
        assert len(modules) == self.NUM_MODULES
        super().__init__(modules, device, enable_finetuning, if_finetuning_finetune_a_copy, random_init_random_seed,
                         output_dim=output_dim)

    @staticmethod
    def forward1(partial_program, inputs, return_logits_as_well=True):
        # partial_program = self.modules
        c_logits = c_outputs = inputs
        for c_pp_layer_idx, c_module in enumerate(partial_program):
            c_logits, c_outputs = c_module(c_outputs, return_logits_as_well=True)

        if return_logits_as_well:
            return c_logits, c_outputs
        return c_outputs

    def forward(self, inputs):
        return self.forward1(self.modules, inputs, return_logits_as_well=True)


class T2Architecture2(NNArchitecture):
    """
    An Architecture, which combines 1 CNN and 1 FC layers
    """
    NUM_MLP_MODULES = 3
    NUM_MODULES = T2Architecture1.NUM_MODULES + NUM_MLP_MODULES

    @staticmethod
    def get_criterion_class():
        return torch.nn.BCEWithLogitsLoss  # return F.binary_cross_entropy_with_logits

    @staticmethod
    def get_num_modules() -> int:
        return T2Architecture2.NUM_MODULES

    @staticmethod
    def get_module_types() -> List[Type[NNModule]]:
        return T2Architecture1.get_module_types() + [
            T1Module_MLP_FCL1, T1Module_MLP_FCL2, T1Module_MLP_FCL3
        ]

    def create_new_module_by_index(self, index) -> NNModule:
        return T2Architecture2.get_module_types()[index]()

    @staticmethod
    def _forward_pp_PT(partial_program, c_inputs):
        raise NotImplementedError()

    @staticmethod
    def get_partial_program_outputs_PT(partial_program: Tuple[NNModule],
                                       data_loader: DataLoader,
                                       device: str) -> Union[np.ndarray, List[np.ndarray]]:
        raise NotImplementedError()

    @staticmethod
    def _forward_pp_HLT(partial_program, c_inputs):
        raise NotImplementedError()

    @staticmethod
    def get_partial_program_outputs_HLT(partial_program: Tuple[NNModule], data_loader: DataLoader, device: str,
                                        use_logits=False) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def get_partial_program_normalized_input_gradients_HLT(partial_program: Tuple[NNModule], data_loader: DataLoader,
                                                           device: str,
                                                           use_logits=True) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def forward_program(inputs, program: Tuple[NNModule], return_logits_as_well=True):
        # inputs is a list of images, represented as a tensor
        # first, split it back into a list
        if type(inputs) == list:
            inputs_list = inputs
        else:
            inputs_list = [torch.squeeze(j, dim=1) for j in torch.split(inputs, 1, dim=1)]
        cnn_fn = lambda x: T2Architecture1.forward1(program[:T2Architecture1.NUM_MODULES], x, return_logits_as_well=False)
        h1_list = list(map(cnn_fn, inputs_list))
        h1_concat = torch.cat(h1_list, dim=1)
        # h1_concat = torch.cat(h1_list, dim=1)

        c_logits, c_outputs = h1_concat, h1_concat
        for m in program[T2Architecture1.NUM_MODULES:]:
            c_logits, c_outputs = m(c_outputs, return_logits_as_well=True)
        return (c_logits, c_outputs) if return_logits_as_well else c_outputs

    @staticmethod
    def get_layers_input_shape(pp_first_module_layer_idx: int):
        # the first item in the list indicates the list's length
        return {
            0: [T1_MODULES_LIST_LENGTH, 10],
            1: [T1_MODULES_LIST_LENGTH, 64],
            2: [T1_MODULES_LIST_LENGTH, 64],
            3: [T1_MODULES_LIST_LENGTH, 64],

            4: (T1_MODULES_FC_IN_DIM,),
            5: (T1_FC_NUM_HIDDEN_UNITS,),
            6: (T1_MODULES_LAST_HIDDEN_STATE_NUM_UNITS,)
        }[pp_first_module_layer_idx]

    def __init__(self, modules: Tuple[Union[NNModule, None]], device: str,
                 enable_finetuning: bool, if_finetuning_finetune_a_copy: bool,
                 random_init_random_seed: Union[int, None],
                 output_dim=None):
        assert len(modules) == self.NUM_MODULES
        super().__init__(modules, device, enable_finetuning, if_finetuning_finetune_a_copy, random_init_random_seed,
                         output_dim=output_dim)

    def forward(self, inputs):
        logits, outputs = self.forward_program(inputs, self.modules, return_logits_as_well=True)
        return logits, outputs
