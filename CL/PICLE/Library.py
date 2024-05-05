from typing import Dict, Type
import numpy as np
from CL.Interface.NNModule import NNModule
from CL.Interface.ModularCL.Library import Library
from CL.PICLE.LibraryItem import PICLELibraryItem
from CL.PICLE.SoftType import SoftType


class PICLELibrary(Library):
    def __init__(self, device: str, soft_type_target_dim: int, num_input_points_stored_per_module):
        super().__init__(device)
        self.soft_type_target_dim = soft_type_target_dim
        self.num_input_points_stored_per_module = num_input_points_stored_per_module

        # a cache, just so I don't have to vstack() every time.
        # not necessary to save/load this, as I'll just "compute" it for every problem.
        self.dict_module_type_to_all_stored_inputs: Dict[Type[NNModule], np.ndarray] = {}

    def add_module(self, name: str, module: NNModule,
                   tr_inputs: np.array, tr_outputs: np.array, performance: float):

        assert module is not None
        c_input_soft_type = SoftType.create_from_data_points(tr_inputs, use_random_projection=True,
                                                             target_dim=self.soft_type_target_dim)

        c_stored_inputs = tr_inputs[:self.num_input_points_stored_per_module]

        lib_item = PICLELibraryItem(name, module, [c_input_soft_type], c_stored_inputs, [performance])
        self._add_item(lib_item)
        self._modules_to_save.append(name)

        # get rid of the cached result, as I'd need to re-compute it, since a new item is stored
        if lib_item.module_type in self.dict_module_type_to_all_stored_inputs.keys():
            del self.dict_module_type_to_all_stored_inputs[lib_item.module_type]

        # Q3: Where to store the functional similarities?
        # A: Store them into a separate dictionary.

    def add_soft_type_to_module(self, name: str, tr_inputs: np.array, tr_outputs: np.array, performance):
        c_lib_item = self._items[name]
        c_input_soft_type = SoftType.create_from_data_points(tr_inputs, use_random_projection=True,
                                                             target_dim=self.soft_type_target_dim)
        c_lib_item.input_soft_types.append(c_input_soft_type)
        c_lib_item.performances.append(performance)
        self._modules_to_save.append(name)

    def get_all_stored_inputs_for_module_type(self, module_type: Type[NNModule]):
        if module_type not in self.dict_module_type_to_all_stored_inputs.keys():
            all_stored_inputs_list = [item.stored_inputs for item in self.items_by_module_type[module_type]]
            all_stored_inputs = np.vstack(all_stored_inputs_list)
            self.dict_module_type_to_all_stored_inputs[module_type] = all_stored_inputs
        return self.dict_module_type_to_all_stored_inputs[module_type]

    @staticmethod
    def _get_lib_item_class():
        return PICLELibraryItem

    def __getitem__(self, name) -> PICLELibraryItem:
        return self._items[name]
