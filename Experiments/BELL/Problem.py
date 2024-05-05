from typing import Tuple, Union
import numpy as np
from CL.Interface.Problem import Problem
from Experiments.BELL.DataCreation.ImageDatasetProviders import *
from Experiments.BELL.NNArchitectures import *
from Experiments.BELL.DataCreation.ProblemDataGenerator import ProblemDataGenerator


class BELLProblem(Problem):
    def get_tr_dataset_tuple(self) -> Tuple[np.array, np.array]:
        tr_in, tr_out = self.tr_dataset_tuple
        return tr_in, tr_out

    def get_IMG_DATA_PROVIDER(self, name,
                              num_tr_datapoints: Union[int, None] = None,
                              tr_shuffle_seed: Union[int, None] = None):
        """
        :param name: {class_name}_{portion}
        :param num_tr_datapoints:
        :param tr_shuffle_seed:
        :return:
        """

        class_name_to_class = {
            "mnist": MNISTDataProvider,
            "fmnist": FMNISTDataProvider,
            "k49": Kuzushiji49DataProvider,
            "emnist": EMNISTDataProvider,
        }
        flatten_inputs = name.endswith("flat")
        if flatten_inputs:
            name = name[:-len("flat")]

        delimiter_idx = name.index('_')
        dataset_name = name[:delimiter_idx]
        class_portion = int(name[delimiter_idx+1:])

        c_class = class_name_to_class[dataset_name]
        return c_class(class_portion, get_test=True, normalize_input=True,
                       num_tr_datapoints=num_tr_datapoints, tr_shuffle_seed=tr_shuffle_seed,
                       flatten_inputs=flatten_inputs)

    def __init__(self,
                 problem_str: Tuple[Union[None, str]],
                 batch_size: int,
                 dataset_folder,
                 random_seed: int,
                 total_num_tr_datapoints: Union[None, int, Tuple[int, int, int]],
                 total_num_val_datapoints: int,
                 total_num_test_datapoints: int,
                 num_id: Union[None, int] = None,
                 _debug_task_same_images_for_tr_and_val=False
                 ):
        self.problem_str = problem_str
        self.random_seed = random_seed
        self.total_num_tr_datapoints = total_num_tr_datapoints
        self.total_num_val_datapoints = total_num_val_datapoints
        self.total_num_test_datapoints = total_num_test_datapoints

        assert len(problem_str) == 2
        assert problem_str[0] is not None   # this might change in the future
        # assert not problem_str[0] is None or not problem_str[1] is None # both can't be None

        self.img_ds_id = self.problem_str[0]
        self.task_id = self.problem_str[1]

        name = '_'.join((str(j) for j in problem_str))
        architecture_class = self._get_architecture_class(problem_str)

        self._debug_task_same_images_for_tr_and_val = _debug_task_same_images_for_tr_and_val

        output_dim = ImageDatasetProvider.NUM_CLASSES_PER_PROBLEM if problem_str[1] is None else 1
        super().__init__(name, dataset_folder, batch_size, architecture_class, num_id=num_id, output_dim=output_dim)

    def create_img_dataset(self):
        assert self.total_num_tr_datapoints is None or type(self.total_num_tr_datapoints) == int

        c_data_provider = self.get_IMG_DATA_PROVIDER(self.img_ds_id,
                                                     num_tr_datapoints=self.total_num_tr_datapoints,
                                                     tr_shuffle_seed=self.random_seed)
        c_ds = c_data_provider.get_images_for_classification()

        tr_tuple = (c_ds['tr']['images'], c_ds['tr']['labels'])
        val_tuple = c_ds['val']['images'], c_ds['val']['labels']
        test_tuple = c_ds['test']['images'], c_ds['test']['labels']

        tr_tuple = (tr_tuple[0], np.argmax(tr_tuple[1], axis=1))
        val_tuple = (val_tuple[0], np.argmax(val_tuple[1], axis=1))
        test_tuple = (test_tuple[0], np.argmax(test_tuple[1], axis=1))

        return tr_tuple, val_tuple, test_tuple

    def create_img_and_task_dataset(self):
        assert self.total_num_tr_datapoints is int or type(self.total_num_tr_datapoints) == tuple
        assert self.total_num_val_datapoints is not None
        assert self.total_num_test_datapoints is not None

        if type(self.total_num_tr_datapoints) == int:
            tr_num_items = self.total_num_tr_datapoints
            tr_num_unique_images = None
            tr_num_unique_patterns = None
        else:
            tr_num_items, tr_num_unique_images, tr_num_unique_patterns = self.total_num_tr_datapoints

        c_img_data_provider = self.get_IMG_DATA_PROVIDER(self.img_ds_id, tr_num_unique_images, self.random_seed)

        pdg = ProblemDataGenerator(self.random_seed, c_img_data_provider,
                                   tr_num_items,
                                   self.total_num_val_datapoints, self.total_num_test_datapoints,
                                   tr_num_unique_patterns,
                                   self._debug_task_same_images_for_tr_and_val)

        tr_tuple, val_tuple, test_tuple = pdg.generate_problem_datasets(self.task_id)
        return tr_tuple, val_tuple, test_tuple

    def _get_architecture_class(self, problem_str: Tuple[Union[None, str]]):
        if problem_str[0].endswith("flat"):
            if problem_str[0] is not None and problem_str[1] is None:
                return T2Architecture1
            elif problem_str[0] is not None and problem_str[1] is not None:
                return T2Architecture2
            else:
                raise NotImplementedError()

        if problem_str[0] is not None and problem_str[1] is None:
            architecture_class = T1Architecture1
        elif problem_str[0] is not None and problem_str[1] is not None:
            architecture_class = T1Architecture2
        else:
            raise NotImplementedError()
        return architecture_class

    def load_datasets(self):
        # First, try to load it
        if super(BELLProblem, self).load_datasets():
            return True

        # If it can't be loaded, generate a dataset
        if self.problem_str[0] is not None and self.problem_str[1] is None:
            ds_tuples = self.create_img_dataset()
        else:
            ds_tuples = self.create_img_and_task_dataset()

        self.tr_dataset_tuple = ds_tuples[0]
        self.val_dataset_tuple = ds_tuples[1]
        self.test_dataset_tuple = ds_tuples[2]

        self.datasets_loaded = True
        return True

