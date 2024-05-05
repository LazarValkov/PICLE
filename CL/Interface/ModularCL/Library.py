from CL.Interface.NNModule import NNModule
from CL.Interface.ModularCL.LibraryItem import LibraryItem
from typing import Dict, List, Type
import numpy as np
import pickle
import glob
import os


class Library:
    def __init__(self, device):
        self.device = device
        self.items_by_module_type: Dict[Type[NNModule], List[LibraryItem]] = {}
        self._items: Dict[str, LibraryItem] = {}
        self._modules_to_save = []
        self.last_loaded_id = None
        self.previous_solutions = []

    def add_module(self, name: str, module: NNModule,
                   tr_inputs: np.ndarray, tr_outputs: np.ndarray, performance: float):
        assert module is not None
        item = LibraryItem(name, module)
        self._add_item(item)
        self._modules_to_save.append(name)

    def _add_item(self, item: LibraryItem):
        """
        DON'T CALL FROM OUTSIDE OF THE CLASS. The item won't be saved.
        """
        if item.module_type not in self.items_by_module_type.keys():
            self.items_by_module_type[item.module_type] = []
        self.items_by_module_type[item.module_type].append(item)
        self._items[item.name] = item

    def __getitem__(self, name) -> LibraryItem:
        return self._items[name]

    @staticmethod
    def _get_lib_item_class():
        return LibraryItem

    def save(self, folder, id=None, save_all_modules=True):
        if not os.path.exists(folder):
            os.makedirs(folder)

        if save_all_modules:
            lib_items_to_save = self._items.values()
        else:
            lib_items_to_save = [self._items[item_name] for item_name in self._modules_to_save]

        for item in lib_items_to_save:
            item.save(folder)

        if id is not None:
            # save a list, detailing all the lib item names in this library
            all_item_names = list(self._items.keys())
            lib_filepath = f"{folder}/lib_{id}.txt"

            # if the file already exists, this should over-write it.
            with open(lib_filepath, 'w') as text_file:
                for item_name in all_item_names:
                    text_file.write(item_name + os.linesep) # os.linesep ends the line
                # text_file.writelines(item_name + os.linesep for item_name in all_item_names)

            # save a list of all solutions found so far
            lib_filepath2 = f"{folder}/lib_{id}_extra_info.bin"
            state_dict_to_save = {"previous_solutions": self.previous_solutions}
            with open(lib_filepath2, 'wb') as f:
                pickle.dump(state_dict_to_save, f, pickle.HIGHEST_PROTOCOL)

    def load(self, folder, id=None):
        if id is None:
            # load all the lib items in the folder
            all_lib_item_filepaths = glob.glob(f"{folder}/*.lib_item")
        else:
            # use the given id to load a list of library items which we will load
            lib_filepath = f"{folder}/lib_{id}.txt"
            with open(lib_filepath, 'r') as text_file:
                all_item_names = text_file.readlines()
                all_lib_item_filepaths = [f"{folder}/{n.strip()}.lib_item" for n in all_item_names]

        lib_item_class = self._get_lib_item_class()
        for c_lib_item_filepath in all_lib_item_filepaths:
            c_name = os.path.basename(c_lib_item_filepath)[:-len(".lib_item")]
            c_lib_item = lib_item_class.load(folder, c_name, self.device)
            self._add_item(c_lib_item)

        # Load the extra information
        if id is not None:
            lib_filepath2 = f"{folder}/lib_{id}_extra_info.bin"
            with open(lib_filepath2, 'rb') as fh:
                loaded_state_dict = pickle.load(fh)
            self.previous_solutions = loaded_state_dict["previous_solutions"]

        self.last_loaded_id = id
