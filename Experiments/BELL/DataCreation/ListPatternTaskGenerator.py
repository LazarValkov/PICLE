import random
import math
import copy
from typing import List
import numpy as np


class ListPatternTaskGenerator:
    # SPLIT_PERCENTAGES = [0.8, 0.1, 0.1]  # 80% for training, 10% for validation, 10% for test

    @staticmethod
    def get_dataset_2d_xor(num_diff_values):
        # usually num_diff_values = 8
        assert num_diff_values % 2 == 0
        half = num_diff_values // 2

        inputs_ll = [(i, j) for i in range(1, num_diff_values + 1) for j in range(1, num_diff_values + 1)]
        labels_l = [(i[0] <= half) != (i[1] <= half) for i in inputs_ll]

        inputs_np = np.array(inputs_ll).astype(np.float32)
        labels_np = np.array(labels_l).astype(np.float32).reshape((-1, 1))

        return inputs_np, labels_np

    @staticmethod
    def get_dataset_2d_circle(num_diff_values, radius=3):
        assert num_diff_values % 2 == 0

        mid_point_np = np.array([[(num_diff_values + 1) / 2, (num_diff_values + 1) / 2]])

        inputs_ll = [(i, j) for i in range(1, num_diff_values + 1) for j in range(1, num_diff_values + 1)]
        inputs_np = np.array(inputs_ll).astype(np.float32)

        inputs_distance_from_center = np.linalg.norm(inputs_ll - mid_point_np, axis=1)
        labels_np = (inputs_distance_from_center < radius).astype(np.float32).reshape((-1, 1))

        return inputs_np, labels_np

    @staticmethod
    def get_dataset_2d_lines(num_diff_values):
        assert num_diff_values % 2 == 0

        A = np.zeros((num_diff_values, num_diff_values))
        # mark some of the rows as 1.
        for i in range(num_diff_values):
            if (i // 2) % 2 == 0:
                A[i] = 1.

        inputs_ll, labels_l = [], []
        for i in range(num_diff_values):
            for j in range(num_diff_values):
                inputs_ll.append(
                    (i + 1, j + 1))  # cos the coordinates in my datasets start with 1 (don't remember why ^^ )
                labels_l.append(A[i, j])

        inputs_np = np.array(inputs_ll).astype(np.float32)
        labels_np = np.array(labels_l).astype(np.float32).reshape((-1, 1))

        return inputs_np, labels_np

    @staticmethod
    def get_dataset_2d_cols(num_diff_values):
        assert num_diff_values % 2 == 0

        A = np.zeros((num_diff_values, num_diff_values))
        # mark some of the rows as 1.
        for i in range(num_diff_values):
            if (i // 2) % 2 == 0:
                A[:, i] = 1.

        inputs_ll, labels_l = [], []
        for i in range(num_diff_values):
            for j in range(num_diff_values):
                inputs_ll.append(
                    (i + 1, j + 1))  # cos the coordinates in my datasets start with 1 (don't remember why ^^ )
                labels_l.append(A[i, j])

        inputs_np = np.array(inputs_ll).astype(np.float32)
        labels_np = np.array(labels_l).astype(np.float32).reshape((-1, 1))

        return inputs_np, labels_np

    @staticmethod
    def get_dataset_2d_X(num_diff_values):
        assert num_diff_values == 8

        indices = [(0, 0), (0, 1), (0, 6), (0, 7), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                   (2, 2), (2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4), (4, 5),
                   (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (6, 0), (6, 1), (6, 6), (6, 7),
                   (7, 0), (7, 7)]

        A = np.zeros((num_diff_values, num_diff_values))
        # mark some of the rows as 1.
        for i in indices:
            A[i] = 1.

        inputs_ll, labels_l = [], []
        for i in range(num_diff_values):
            for j in range(num_diff_values):
                inputs_ll.append(
                    (i + 1, j + 1))  # cos the coordinates in my datasets start with 1 (don't remember why ^^ )
                labels_l.append(A[i, j])

        inputs_np = np.array(inputs_ll).astype(np.float32)
        labels_np = np.array(labels_l).astype(np.float32).reshape((-1, 1))

        return inputs_np, labels_np

    @staticmethod
    def get_dataset_2d_SQUARE(num_diff_values):
        assert num_diff_values == 8

        indices = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 0), (2, 0), (3, 0), (4, 0),
                   (5, 0), (6, 0), (7, 0),
                   (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5),
                   (7, 6), (3, 3), (3, 4), (4, 3), (4, 4)]

        A = np.zeros((num_diff_values, num_diff_values))
        # mark some of the rows as 1.
        for i in indices:
            A[i] = 1.

        inputs_ll, labels_l = [], []
        for i in range(num_diff_values):
            for j in range(num_diff_values):
                inputs_ll.append(
                    (i + 1, j + 1))  # cos the coordinates in my datasets start with 1 (don't remember why ^^ )
                labels_l.append(A[i, j])

        inputs_np = np.array(inputs_ll).astype(np.float32)
        labels_np = np.array(labels_l).astype(np.float32).reshape((-1, 1))

        return inputs_np, labels_np


    task_to_ds_gen_fn = {
        # the pattern describes the indices of a random selection of classes
        "2dXOR": get_dataset_2d_xor.__func__,
        "2dCircle": get_dataset_2d_circle.__func__,
        "2dX": get_dataset_2d_X.__func__,
        "2dSquare": get_dataset_2d_SQUARE.__func__,

        "2dLines": get_dataset_2d_lines.__func__,

        "2dRows": get_dataset_2d_lines.__func__,
        "2dCols": get_dataset_2d_cols.__func__,
    }
    task_and_perm_to_seed_dict = {
        ("2dXOR", "perm1"): 9,
        ("2dXOR", "perm2"): 10,
        ("2dXOR", "perm3"): 11,

        ("2dCircle", "perm1"): 12,
        ("2dCircle", "perm2"): 13,
        ("2dCircle", "perm3"): 14,

        ("2dX", "perm1"): 15,
        ("2dX", "perm2"): 16,
        ("2dX", "perm3"): 17,

        ("2dSquare", "perm1"): 18,
        ("2dSquare", "perm2"): 19,
        ("2dSquare", "perm3"): 20,
    }
    perm_to_seed_dict = {
        "perm1": 111,
        "perm2": 112,
        "perm3": 113,
    }
    task_to_split_seed = {
        "2dLines": 819,
        "2dRows": 819,
        "2dCols": 819,

        "2dXOR": 3894,
        "2dCircle": 84
    }

    def split_tr_val_test(self, ds_in, ds_out, tr_perc=80):
        # SPLIT_RANDOM_SEED = 2

        _, c_label_counts = np.unique(ds_out, return_counts=True)
        assert c_label_counts[0] == c_label_counts[1]

        num_all = ds_in.shape[0]
        num_val_pos = num_val_neg = num_test_pos = num_test_neg = int(((1. - tr_perc / 100) / 4) * num_all)
        # num_tr = num_all - (num_val_pos + num_val_neg + num_test_pos + num_test_neg)

        ds_in_pos, ds_out_pos = ds_in[ds_out.reshape((-1,)) == 1.], ds_out[ds_out.reshape((-1,)) == 1.]
        ds_in_neg, ds_out_neg = ds_in[ds_out.reshape((-1,)) == 0.], ds_out[ds_out.reshape((-1,)) == 0.]

        indices_pos = list(range(ds_in_pos.shape[0]))
        indices_neg = list(range(ds_in_neg.shape[0]))
        c_rndm = random.Random(self.task_ds_split_random_seed)
        c_rndm.shuffle(indices_pos)
        c_rndm.shuffle(indices_neg)

        val_pos_indices = indices_pos[:num_val_pos]
        val_neg_indices = indices_neg[:num_val_neg]
        test_pos_indices = indices_pos[num_val_pos:num_val_pos + num_test_pos]
        test_neg_indices = indices_neg[num_val_neg:num_val_neg + num_test_neg]
        train_pos_indices = indices_pos[num_val_pos + num_test_pos:]
        train_neg_indices = indices_neg[num_val_neg + num_test_neg:]

        val_in_pos, val_out_pos = ds_in_pos[val_pos_indices], ds_out_pos[val_pos_indices]
        val_in_neg, val_out_neg = ds_in_neg[val_neg_indices], ds_out_neg[val_neg_indices]
        val_in, val_out = np.vstack((val_in_pos, val_in_neg)), np.vstack((val_out_pos, val_out_neg))

        test_in_pos, test_out_pos = ds_in_pos[test_pos_indices], ds_out_pos[test_pos_indices]
        test_in_neg, test_out_neg = ds_in_neg[test_neg_indices], ds_out_neg[test_neg_indices]
        test_in, test_out = np.vstack((test_in_pos, test_in_neg)), np.vstack((test_out_pos, test_out_neg))

        tr_in_pos, tr_out_pos = ds_in_pos[train_pos_indices], ds_out_pos[train_pos_indices]
        tr_in_neg, tr_out_neg = ds_in_neg[train_neg_indices], ds_out_neg[train_neg_indices]
        tr_in, tr_out = np.vstack((tr_in_pos, tr_in_neg)), np.vstack((tr_out_pos, tr_out_neg))

        return (tr_in, tr_out), (val_in, val_out), (test_in, test_out)

    def __init__(self, task_id, classes: List[int], random_obj, tr_num_unique_patterns):
        self.classes = classes
        self.random_obj = random_obj  # random.Random(random_seed)

        self.task_id = task_id
        pattern_name = task_id[:task_id.index('_')]
        permutation_name = task_id[task_id.index('_') + 1:]
        self.ds_gen_fn = self.task_to_ds_gen_fn[pattern_name]
        # self.task_ds_split_random_seed = self.task_to_split_seed[pattern_name]

        # Figure out the class - integer value mapping
        classes_diff_numerical_values = list(range(1, len(classes)+1))
        if permutation_name != "identity":
            c_perm_seed = self.task_and_perm_to_seed_dict[(pattern_name, permutation_name)]
            # c_perm_seed = self.perm_to_seed_dict[permutation_name]
            c_perm_random = random.Random(c_perm_seed)
            c_perm_random.shuffle(classes_diff_numerical_values)

        self.class_to_num_value = {c: classes_diff_numerical_values[i] for i, c in enumerate(classes)}
        self.num_value_to_class = {classes_diff_numerical_values[i]: c for i, c in enumerate(classes)}

        self.ds_in_np, self.ds_labels_np = self.ds_gen_fn(len(classes))

        # re-map the original values to classes
        self.ds_in_remapped_to_classes_np = np.copy(self.ds_in_np)
        for row in range(self.ds_in_remapped_to_classes_np.shape[0]):
            for col in range(self.ds_in_remapped_to_classes_np.shape[1]):
                c_val = self.ds_in_remapped_to_classes_np[row, col]
                c_class = self.num_value_to_class[c_val]
                self.ds_in_remapped_to_classes_np[row, col] = c_class

        # ATM: Use the same tr, val, test
        self.tr_inputs_np, self.tr_targets_np = self.ds_in_remapped_to_classes_np, self.ds_labels_np
        self.val_inputs_np, self.val_targets_np = self.ds_in_remapped_to_classes_np, self.ds_labels_np
        self.test_inputs_np, self.test_targets_np = self.ds_in_remapped_to_classes_np, self.ds_labels_np

        #split = self.split_tr_val_test(self.ds_in_remapped_to_classes_np, self.ds_labels_np)
        #self.tr_inputs_np, self.tr_targets_np = split[0][0], split[0][1]
        #self.val_inputs_np, self.val_targets_np = split[1][0], split[1][1]
        #self.test_inputs_np, self.test_targets_np = split[2][0], split[2][1]

        if tr_num_unique_patterns is not None:
            assert tr_num_unique_patterns <= self.tr_inputs_np.shape[0]
            self.tr_inputs_np = self.tr_inputs_np[:tr_num_unique_patterns]
            self.tr_targets_np = self.tr_targets_np[:tr_num_unique_patterns]

    def get_random_datasset(self, inputs_np, targets_np, num_items):
        indices_left_to_select = num_items

        selected_inputs = []
        selected_labels = []

        while indices_left_to_select >= inputs_np.shape[0]:
            selected_inputs.append(inputs_np)
            selected_labels.append(targets_np)
            indices_left_to_select -= inputs_np.shape[0]

        if indices_left_to_select > 0:
            c_selected_indices = self.random_obj.sample(list(range(inputs_np.shape[0])), indices_left_to_select)
            c_selected_inputs_np = inputs_np[c_selected_indices]
            c_selected_targets_np = targets_np[c_selected_indices]

            selected_inputs.append(c_selected_inputs_np)
            selected_labels.append(c_selected_targets_np)

        selected_inputs_np = np.vstack(selected_inputs)
        selected_labels_np = np.vstack(selected_labels)
        return selected_inputs_np, selected_labels_np

    def get_random_tr_dataset(self, num_items):
        return self.get_random_datasset(self.tr_inputs_np, self.tr_targets_np, num_items)

    def get_random_val_dataset(self, num_items):
        return self.get_random_datasset(self.val_inputs_np, self.val_targets_np, num_items)

    def get_random_test_dataset(self, num_items):
        return self.get_random_datasset(self.test_inputs_np, self.test_targets_np, num_items)
