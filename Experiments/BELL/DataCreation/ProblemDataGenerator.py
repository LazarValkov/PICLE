import numpy as np
import random

from Experiments.BELL.DataCreation.ImageDatasetProviders import ImageDatasetProvider
from Experiments.BELL.DataCreation.ListPatternTaskGenerator import ListPatternTaskGenerator
from typing import List
from typing import Union, Tuple


class ProblemDataGenerator:
    def __init__(self, random_seed: int, img_ds_provider: ImageDatasetProvider,
                 num_dp_tr: int,
                 num_dp_val: int,
                 num_dp_test: int,
                 tr_num_unique_patterns: Union[None, int],
                 _debug_task_same_images_for_tr_and_val: bool = False):
        self.random_obj = random.Random(random_seed)
        self.img_ds_provider = img_ds_provider
        self.num_dp_tr, self.num_dp_val, self.num_dp_test = num_dp_tr, num_dp_val, num_dp_test
        self.tr_num_unique_patterns = tr_num_unique_patterns

        self._debug_task_same_images_for_tr_and_val = _debug_task_same_images_for_tr_and_val

    def replace_w_images(self, task_inputs_np, ds_images_dict):
        # replace the classes with images
        num_dp = task_inputs_np.shape[0]
        list_len = task_inputs_np.shape[1]

        num_ch = self.img_ds_provider.get_image_channels_num()
        img_size = self.img_ds_provider.get_image_size()

        if self.img_ds_provider.flatten_inputs:
            dim = num_ch * img_size * img_size
            c_inputs_np = np.zeros((num_dp, list_len, dim), dtype=np.float32)
        else:
            c_inputs_np = np.zeros((num_dp, list_len, num_ch, img_size, img_size), dtype=np.float32)

        c_dict_class_to_indices_left = {}

        for r in range(num_dp):
            for c in range(list_len):
                c_class = task_inputs_np[r, c]
                # c_img = self.img_ds_provider.get_random_image(c_class, ds_images_dict, self.random_obj)
                c_img = self.img_ds_provider.get_random_image_without_replacement(c_class, ds_images_dict,
                                                                                  c_dict_class_to_indices_left,
                                                                                  self.random_obj)
                c_inputs_np[r, c] = c_img

        return c_inputs_np

    def generate_problem_datasets(self, task_id: str):
        classes = list(range(self.img_ds_provider.get_num_classes()))
        c_list_gen = ListPatternTaskGenerator(task_id, classes, self.random_obj, self.tr_num_unique_patterns)

        tr_task_in, tr_task_targets = c_list_gen.get_random_tr_dataset(self.num_dp_tr)
        tr_inputs = self.replace_w_images(tr_task_in, self.img_ds_provider.ds_images_tr)

        val_task_in, val_task_targets = c_list_gen.get_random_val_dataset(self.num_dp_val)
        if self._debug_task_same_images_for_tr_and_val:
            val_inputs = self.replace_w_images(val_task_in, self.img_ds_provider.ds_images_tr)
        else:
            val_inputs = self.replace_w_images(val_task_in, self.img_ds_provider.ds_images_val)

        test_task_in, test_task_targets = c_list_gen.get_random_test_dataset(self.num_dp_test)
        test_inputs = self.replace_w_images(test_task_in, self.img_ds_provider.ds_images_test)

        return (tr_inputs, tr_task_targets), (val_inputs, val_task_targets), (test_inputs, test_task_targets)
