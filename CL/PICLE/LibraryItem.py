from CL.Interface.NNModule import NNModule
from CL.Interface.ModularCL.LibraryItem import LibraryItem
from CL.PICLE.SoftType import SoftType
import numpy as np
from typing import Type, Dict, List
import os


class PICLELibraryItem(LibraryItem):
    def __init__(self, name: str, module: NNModule,
                 input_soft_types: List[SoftType],
                 stored_inputs: np.ndarray,
                 performances: List[float]
                 ):
        # the approximated input distribution of a module is referred to as an input soft type
        super().__init__(name, module)
        assert type(input_soft_types) == list
        self.input_soft_types = input_soft_types
        self.performances = performances
        self.stored_inputs = stored_inputs

    @staticmethod
    def _load_get_init_parameters(folder: str, name: str, device: str):
        load_lib_item_dict = super(PICLELibraryItem, PICLELibraryItem)._load_get_init_parameters(folder, name, device)

        # load the stored inputs
        stored_inputs_filepath = f"{folder}/{name}_stored_inputs.npy"
        load_lib_item_dict["stored_inputs"] = np.load(stored_inputs_filepath)

        return load_lib_item_dict

    @staticmethod
    def load(folder: str, name: str, device: str):
        return PICLELibraryItem(**PICLELibraryItem._load_get_init_parameters(folder, name, device))

    def get_serializable_dict(self):
        dict = super(PICLELibraryItem, self).get_serializable_dict()
        dict["input_soft_types"] = self.input_soft_types
        dict["performances"] = self.performances
        return dict

    def save(self, folder: str):
        if not os.path.exists(folder):
            os.makedirs(folder)

        # save the stored input items
        stored_inputs_filepath = f"{folder}/{self.name}_stored_inputs.npy"
        np.save(stored_inputs_filepath, self.stored_inputs)

        # save the rest
        super(PICLELibraryItem, self).save(folder)

