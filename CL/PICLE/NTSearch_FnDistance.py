from typing import Dict, Union, Tuple, List, Type
import numpy as np
import torch
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter
from abc import abstractmethod, ABC
import random
import math
from CL.PICLE.Library import PICLELibrary
from CL.Interface.NNArchitecture import NNArchitecture
from CL.Interface.NNModule import NNModule
import scipy


class _CustomSingleProcessDataLoaderIter(_SingleProcessDataLoaderIter):
    def _next_data(self):
        return (list(super()._next_data()),)


class ListDataLoader(torch.utils.data.DataLoader):
    def __init__(self, list_inputs_np: List[np.ndarray], batch_size: int, shuffle=False):
        list_inputs_torch = [torch.from_numpy(a) for a in list_inputs_np]
        ds_tds = torch.utils.data.TensorDataset(*list_inputs_torch)
        super().__init__(ds_tds, batch_size=batch_size, shuffle=shuffle)

    def _get_iterator(self):
        return _CustomSingleProcessDataLoaderIter(self)


class BaseFunctionalSimilarityCalculator(ABC):
    # base class which can be extended to compute different distances/similarities between functions
    def __init__(self, lib: PICLELibrary,
                 nn_architecture_class: Type[NNArchitecture],
                 pp_first_module_layer_idx: int,
                 device: str):
        self.lib = lib
        self.nn_architecture_class = nn_architecture_class
        self.device = device

        first_module_type = nn_architecture_class.get_module_types()[pp_first_module_layer_idx]
        # use first_module_type and the library to accumulate the current dataset
        c_inputs_for_eval = self.lib.get_all_stored_inputs_for_module_type(first_module_type)

        c_inputs_single_item_shape = c_inputs_for_eval[0].shape

        # check if the stored inputs are of a sub-type.
        # For instance, we have stored Image, but now we need List<Image>
        c_layers_input_shape = nn_architecture_class.get_layers_input_shape(pp_first_module_layer_idx)
        if type(c_layers_input_shape) == list:
            c_list_len = c_layers_input_shape[0]
            list_item_shape = tuple(c_layers_input_shape[1:])
            assert list_item_shape == c_inputs_single_item_shape

            self.c_inputs_for_eval = self.generate_random_list_using_inputs(c_list_len, c_inputs_for_eval)
            self.inputs_for_eval_ds_loader = ListDataLoader(self.c_inputs_for_eval, batch_size=64, shuffle=False)
        else:
            if c_inputs_single_item_shape != c_layers_input_shape:
                # assuming that c_layers_input_shape is something like [A, B, C, D]
                # while the c_inputs_for_eval's shape is something like [C, D]
                # i.e. the latter's shape has the same integers as the last dimensions of the former's shape
                assert len(c_layers_input_shape) > len(c_inputs_single_item_shape)
                assert c_layers_input_shape[-len(c_inputs_single_item_shape):] == c_inputs_single_item_shape

                c_inputs_for_eval = self.convert_inputs_using_random_sampling(c_inputs_for_eval, c_layers_input_shape)

            self.inputs_for_eval = c_inputs_for_eval
            self.inputs_for_eval_ds_loader = self._get_data_loader(self.inputs_for_eval)

    @staticmethod
    def generate_random_list_using_inputs(list_length, inputs_np):
        rnd = random.Random(3)
        resulting_list = [ [] for _ in range(list_length)]
        num_inputs_to_generate = inputs_np.shape[0]
        for _ in range(num_inputs_to_generate):
            for j in range(list_length):
                c_random_item_ids = rnd.randint(0, inputs_np.shape[0]-1)
                c_random_item = inputs_np[c_random_item_ids]
                c_random_item = np.expand_dims(c_random_item, axis=0) # prep for vstack later
                resulting_list[j].append(c_random_item)

        resulting_list_new = [np.vstack(li) for li in resulting_list]
        return resulting_list_new

    @staticmethod
    def convert_inputs_using_random_sampling(og_inputs_np, target_shape):
        og_inputs_single_item_shape = og_inputs_np[0].shape

        target_num_random_inputs = og_inputs_np.shape[0]
        new_inputs_shape = (target_num_random_inputs,) + target_shape
        new_inputs = np.zeros(new_inputs_shape, dtype=og_inputs_np.dtype)

        num_random_samples_needed = math.prod(target_shape[:-len(og_inputs_single_item_shape)])

        def get_current_index(i, shape):
            c_index = ()
            prev_num_el = 1
            for s in shape[::-1]:
                j = (i // prev_num_el) % s
                # c_index.insert(j, 0)
                c_index = (j,) + c_index

                prev_num_el *= s
            return c_index

        rnd = random.Random(3)
        for i in range(num_random_samples_needed):
            c_idx = get_current_index(i, new_inputs_shape[:-len(og_inputs_single_item_shape)])
            c_rndm_item_idx = rnd.randint(0, og_inputs_np.shape[0]-1)
            c_rndm_item = og_inputs_np[c_rndm_item_idx]

            new_inputs[c_idx] = c_rndm_item

        return new_inputs

    def _get_data_loader(self, np_inputs, batch_size=64):
        ds_in = torch.from_numpy(np_inputs)
        ds_tds = torch.utils.data.TensorDataset(ds_in)
        ds_loader = torch.utils.data.DataLoader(ds_tds, batch_size=batch_size, shuffle=False)
        return ds_loader

    @abstractmethod
    def cache_partial_program(self, partial_program: Tuple[NNModule],
                                    partial_program_strs: Union[Tuple, Tuple[str]]):
        pass


class FunctionalDistanceCalculator(BaseFunctionalSimilarityCalculator):
    def __init__(self, lib: PICLELibrary, nn_architecture_class: Type[NNArchitecture],
                 pp_first_module_layer_idx: int, device: str,
                 distance_measure:str = "l2"):
        super().__init__(lib, nn_architecture_class, pp_first_module_layer_idx, device)
        assert distance_measure == "l2"
        self.distance_measure = distance_measure
        self.prog_strs_to_output_logits_cache: Dict[Union[Tuple, Tuple[str]], np.ndarray] = {}

    def compute_outputs(self, partial_program: Tuple[NNModule],
                        partial_program_strs: Union[Tuple, Tuple[str]]):

        assert partial_program_strs not in self.prog_strs_to_output_logits_cache.keys()

        outputs = self.nn_architecture_class.get_partial_program_outputs_HLT(
            partial_program, self.inputs_for_eval_ds_loader, self.device,
            use_logits = True)

        self.prog_strs_to_output_logits_cache[partial_program_strs] = outputs
        return outputs

    def get_outputs(self, pp_strs: Union[Tuple, Tuple[str]]):
        # check if they have been cached
        if pp_strs in self.prog_strs_to_output_logits_cache.keys():
            return self.prog_strs_to_output_logits_cache[pp_strs]

        # if not, compute
        pp = tuple(self.lib[module_name].module for module_name in pp_strs)
        return self.compute_outputs(pp, pp_strs)

    def cache_partial_program(self, partial_program: Tuple[NNModule],
                              partial_program_strs: Union[Tuple, Tuple[str]]):
        return self.compute_outputs(partial_program, partial_program_strs)

    def get_fn_distance_between_two_partial_programs(self,
                                                       pp1_strs: Union[Tuple, Tuple[str]],
                                                       pp2_strs: Union[Tuple, Tuple[str]]):
        logits1 = self.get_outputs(pp1_strs)
        logits2 = self.get_outputs(pp2_strs)

        # approximate the l2 distance between the functions, based on the computed logits
        distance = np.sqrt(np.power(logits1.reshape((-1)) - logits2.reshape((-1,)), 2).mean())
        return distance