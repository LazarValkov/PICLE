import torch
from CL.Interface.NNModule import NNModule
from typing import Dict, List, Type
import numpy as np
import pickle
import glob
import os


class LibraryItem:
    def __init__(self, name: str, module: NNModule):
        self.name = name
        self.module_type = type(module)
        self.module = module

    def get_serializable_dict(self):
        rdict = {}
        if self.module.needs_output_dim():
            rdict["output_dim"] = self.module.output_dim
        rdict["name"] = self.name
        rdict["module_type"] = self.module_type
        return rdict
        # "name": , "module_type":

    def save(self, folder: str):
        lib_item_dict = self.get_serializable_dict()
        lib_item_dict_filepath = f"{folder}/{self.name}.lib_item"

        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(lib_item_dict_filepath, 'wb') as f:
            pickle.dump(lib_item_dict, f, pickle.HIGHEST_PROTOCOL)

        # save the module
        module_state_dict = self.module.state_dict()
        # put it on the CPU before saving
        for k, v in module_state_dict.items():
            module_state_dict[k] = v.cpu()
        module_state_dict_filepath = f"{folder}/{self.name}.pt"
        torch.save(module_state_dict, module_state_dict_filepath)

    @staticmethod
    def load_lib_item_dict(folder, name):
        # load the dictionary
        lib_item_dict_filepath = f"{folder}/{name}.lib_item"
        with open(lib_item_dict_filepath, 'rb') as fh:
            lib_item_dict = pickle.load(fh)
        return lib_item_dict

    @staticmethod
    def load_module(folder: str, name: str, module_class: type(NNModule), device: str, output_dim=None):
        # load the module
        module_state_dict_filepath = f"{folder}/{name}.pt"
        if output_dim is None:
            module = module_class()
        else:
            module = module_class(output_dim=output_dim)
        module.load_state_dict(torch.load(module_state_dict_filepath, map_location=torch.device(device)))
        module = module.to(device)
        module.eval()
        return module

    @staticmethod
    def _load_get_init_parameters(folder: str, name: str, device: str):
        load_lib_item_dict = LibraryItem.load_lib_item_dict(folder, name)

        module_class = load_lib_item_dict["module_type"]
        del load_lib_item_dict["module_type"]

        if "output_dim" in load_lib_item_dict.keys():
            output_dim = load_lib_item_dict["output_dim"]
            del load_lib_item_dict["output_dim"]
        else:
            output_dim = None

        module = LibraryItem.load_module(folder, name, module_class, device, output_dim)
        load_lib_item_dict["module"] = module

        return load_lib_item_dict

    @staticmethod
    def load(folder: str, name: str, device: str):
        init_parameters = LibraryItem._load_get_init_parameters(folder, name, device)
        return LibraryItem(**init_parameters)
