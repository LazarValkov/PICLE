from abc import ABC, abstractmethod
from typing import Tuple, Type
import numpy as np
import torch
from CL.Interface.NNArchitecture import NNArchitecture
# from LML.NNArchitecture import NNArchitecture


class Problem(ABC):
    """
    Defines the interface for a supervised learning problem
    """
    def __init__(self, name, dataset_folder, batch_size: int,
                 architecture_class: Type[NNArchitecture],
                 num_id: int,
                 output_dim=None):
        """
        :param num_id: Specifies the order in the problem sequence in which this problem appears
        """
        self.name = name
        self.dataset_folder = dataset_folder
        self.batch_size = batch_size
        self.architecture_class = architecture_class
        self.num_id = num_id

        self.datasets_loaded = False
        self.tr_dataset_tuple = None
        self.val_dataset_tuple = None
        self.test_dataset_tuple = None

        self.output_dim = output_dim

    def unload_datasets(self):
        assert self.datasets_loaded

        self.tr_dataset_tuple = None
        self.val_dataset_tuple = None
        self.test_dataset_tuple = None

        self.datasets_loaded = False

    @abstractmethod
    def load_datasets(self) -> bool:
        """
        :return: True if successful
        """
        pass

    @abstractmethod
    def get_tr_dataset_tuple(self) -> Tuple[np.array, np.array]:
        pass

    def get_data_loader(self, ds_in, ds_out, shuffle):
        ds_in, ds_out = torch.from_numpy(ds_in), torch.from_numpy(ds_out)
        ds_tds = torch.utils.data.TensorDataset(ds_in, ds_out)

        if shuffle:
            # fixing the generator for reproducibility
            g = torch.Generator()
            g.manual_seed(0)
            ds_loader = torch.utils.data.DataLoader(ds_tds, batch_size=self.batch_size, shuffle=shuffle, generator=g)
        else:
            ds_loader = torch.utils.data.DataLoader(ds_tds, batch_size=self.batch_size, shuffle=shuffle)

        return ds_loader

    def get_tr_data_loader(self):
        if not self.datasets_loaded:
            self.load_datasets()
        tr_ds_tuple = self.get_tr_dataset_tuple()

        return self.get_data_loader(tr_ds_tuple[0], tr_ds_tuple[1], shuffle=True)

    def get_val_data_loader(self):
        if not self.datasets_loaded:
            self.load_datasets()
        return self.get_data_loader(self.val_dataset_tuple[0], self.val_dataset_tuple[1], shuffle=False)

    def get_test_data_loader(self):
        if not self.datasets_loaded:
            self.load_datasets()
        return self.get_data_loader(self.test_dataset_tuple[0], self.test_dataset_tuple[1], shuffle=False)
