# from Experiments.Setting6.Problems import ProblemSequence_Setting6
# from Experiments.BELL.Problem import BELLProblem
from Experiments.BELL.ProblemSequence import BELLProblemSequence
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from Experiments.BELL.DataCreation.ImageDatasetProviders import *


class BELLProblem_SequencesGenerator:
    def __init__(self,
                 batch_size,
                 num_tr_dp_max_perf: Tuple[int, Optional[int], Optional[int]],
                 num_tr_dp_lower_perf: Tuple[int, Optional[int], Optional[int]],
                 num_tr_dp_lower_perf_hlt: Tuple[int, Optional[int], Optional[int]],
                 num_tr_dp_lower_perf_few: Tuple[int, Optional[int], Optional[int]],
                 num_val: int, num_test: int):
        self.batch_size = batch_size
        self.num_tr_dp_max_perf = num_tr_dp_max_perf
        self.num_tr_dp_lower_perf = num_tr_dp_lower_perf
        self.num_tr_dp_lower_perf_hlt = num_tr_dp_lower_perf_hlt
        self.num_tr_dp_lower_perf_few = num_tr_dp_lower_perf_few
        self.num_val = num_val
        self.num_test = num_test

    @staticmethod
    def get_all_image_dataset_ids() -> List[str]:
        class_name_to_class = {
            "mnist": MNISTDataProvider,
            "fmnist": FMNISTDataProvider,
            "k49": Kuzushiji49DataProvider,
            "emnist": EMNISTDataProvider
        }
        all_image_dataset_ids = []
        for c_class_name, c_class in class_name_to_class.items():
            num_class_portions = len(c_class._get_full_class_map()) // ImageDatasetProvider.NUM_CLASSES_PER_PROBLEM
            for p in range(1, num_class_portions + 1):
                c_name = f"{c_class_name}_{p}"
                all_image_dataset_ids.append(c_name)
        return all_image_dataset_ids

    @staticmethod
    def get_tasks_ids(identity_only=False) -> List[str]:
        task_ids = ["2dXOR", "2dCircle", "2dX", "2dSquare"]
        if identity_only:
            return [f"{ti}_identity" for ti in task_ids]

        permutations = ["identity", "perm1", "perm2", "perm3"]
        return [f"{ti}_{perm}" for ti in task_ids for perm in permutations]

    @classmethod
    def get_all_problems(cls) -> List[Tuple[str, str]]:
        return [(i, t) for i in cls.get_all_image_dataset_ids() for t in cls.get_tasks_ids()]

    def get_S_pl(self, c_random_seed: int, length=6):
        # S_1^{100%}, S_2^{100%}, S_3^{100%}, S_4^{100%}, S_5^{100%}, S_6^{100%}
        # Evaluating the plasticity. all problems have different images and tasks.
        c_random = random.Random(c_random_seed)

        different_image_ids_selected = c_random.sample(self.get_all_image_dataset_ids(), length)
        different_tasks_selected = c_random.sample(self.get_tasks_ids(), length)
        problems_selected_str = [(different_image_ids_selected[i], different_tasks_selected[i]) for i in range(length)]

        problem_seq = BELLProblemSequence(problems_selected_str,
                                   self.batch_size, None,
                                   c_random_seed,
                                   self.num_tr_dp_max_perf,
                                   self.num_val,
                                   self.num_test)

        return problem_seq

    def get_S_minus(self, c_random_seed: int, length=6):
        # S_1^{100%}, S_2^{60%}, S_3^{60%}, S_4^{60%}, S_5^{60%}, S_1^{60%}
        # Evaluating perceptual and task transfer.
        c_random = random.Random(c_random_seed)

        different_image_ids_selected = c_random.sample(self.get_all_image_dataset_ids(), length - 1)
        different_tasks_selected = c_random.sample(self.get_tasks_ids(), length - 1)
        problems_selected_str = [(different_image_ids_selected[i], different_tasks_selected[i]) for i in
                                 range(length - 1)]
        problems_selected_str += [problems_selected_str[0]]

        num_tr = [self.num_tr_dp_max_perf] + [self.num_tr_dp_lower_perf] * (length - 1)

        problem_seq = BELLProblemSequence(problems_selected_str,
                                   self.batch_size, None,
                                   c_random_seed,
                                   num_tr,
                                   self.num_val,
                                   self.num_test)

        return problem_seq

    def get_S_plus(self, c_random_seed: int, length=6):
        # S_1^{60%}, S_2^{60%}, S_3^{60%}, S_4^{60%}, S_5^{60%}, S_1^{100%}
        # Evaluating avoiding nagative transfer and backward transfer
        c_random = random.Random(c_random_seed)

        different_image_ids_selected = c_random.sample(self.get_all_image_dataset_ids(), length - 1)
        different_tasks_selected = c_random.sample(self.get_tasks_ids(), length - 1)
        problems_selected_str = [(different_image_ids_selected[i], different_tasks_selected[i]) for i in
                                 range(length - 1)]
        problems_selected_str += [problems_selected_str[0]]

        num_tr = [self.num_tr_dp_lower_perf] * (length - 1) + [self.num_tr_dp_max_perf]

        problem_seq = BELLProblemSequence(problems_selected_str,
                                               self.batch_size, None,
                                               c_random_seed,
                                               num_tr,
                                               self.num_val,
                                               self.num_test)

        return problem_seq

    def get_S_in(self, c_random_seed: int, length=6):
        # S_1^{100%}, S_2^{100%}, S_3^{100%}, S_4^{100%}, S_5^{100%}, S_6=(D_6, T_1)^{60%}
        # Evaluating high-level transfer
        c_random = random.Random(c_random_seed)

        different_image_ids_selected = c_random.sample(self.get_all_image_dataset_ids(), length)
        different_tasks_selected = c_random.sample(self.get_tasks_ids(), length - 1)
        different_tasks_selected += [different_tasks_selected[0]]

        problems_selected_str = [(different_image_ids_selected[i], different_tasks_selected[i]) for i in range(length)]

        num_tr = [self.num_tr_dp_max_perf] * (length - 1) + [self.num_tr_dp_lower_perf_hlt]

        problem_seq = BELLProblemSequence(problems_selected_str,
                                               self.batch_size, None,
                                               c_random_seed,
                                               num_tr,
                                               self.num_val,
                                               self.num_test)
        return problem_seq

    def get_S_sp(self, c_random_seed: int, length=6):
        # same code as before, just making the last problem flat
        c_random = random.Random(c_random_seed)

        different_image_ids_selected = c_random.sample(self.get_all_image_dataset_ids(), length)
        different_tasks_selected = c_random.sample(self.get_tasks_ids(), length - 1)
        different_tasks_selected += [different_tasks_selected[0]]

        # make the last flat
        different_image_ids_selected[-1] = different_image_ids_selected[-1] + "flat"

        problems_selected_str = [(different_image_ids_selected[i], different_tasks_selected[i]) for i in range(length)]

        num_tr = [self.num_tr_dp_max_perf] * (length - 1) + [self.num_tr_dp_lower_perf_hlt]

        problem_seq = BELLProblemSequence(problems_selected_str,
                                               self.batch_size, None,
                                               c_random_seed,
                                               num_tr,
                                               self.num_val,
                                               self.num_test)
        return problem_seq

    def get_S_long(self, c_random_seed: int, length=100):
        """
        A: FULL
        B: LLT
        C: HLT
        D: FEW-SHOT TRANSFER

        [1-50]    :     {2/4 A,  1/4 B,  1/4 C       }
        [51:100]:       {        2/5 B,  2/5 C, 1/5 D}
        """
        assert length == 100

        # Evaluating high-level transfer
        c_random = random.Random(c_random_seed)

        all_img_ids = self.get_all_image_dataset_ids()
        all_task_ids = self.get_tasks_ids()

        image_ids_selected = c_random.choices(all_img_ids, k=length)
        task_ids_selected = c_random.choices(all_task_ids, k=length)
        problems_selected_str = [(image_ids_selected[i], task_ids_selected[i]) for i in range(length)]

        A, B, C, D = self.num_tr_dp_max_perf, self.num_tr_dp_lower_perf, \
                     self.num_tr_dp_lower_perf_hlt, self.num_tr_dp_lower_perf_few

        num_tr_1st_half = c_random.choices([A, B, C], [0.5, 0.25, 0.25], k=length // 2)
        num_tr_2nd_half = c_random.choices([B, C, D], [0.4, 0.4, 0.2], k=length // 2)
        num_tr = num_tr_1st_half + num_tr_2nd_half

        problem_seq = BELLProblemSequence(problems_selected_str,
                                               self.batch_size, None,
                                               c_random_seed,
                                               num_tr,
                                               self.num_val,
                                               self.num_test)
        return problem_seq

    def get_S_few(self, c_random_seed: int, length=6):
        # P1=(D1, h1), P2=(D2, h2), P3^-, P4^- = (D1, h1, g4), P5^-, P6^-- = (D2, h2, g4)
        # 4 different domains  /h
        # 3 different g

        assert length == 6

        # Evaluating few-shot transfer
        c_random = random.Random(c_random_seed)

        different_image_ids_selected = c_random.sample(self.get_all_image_dataset_ids(), 4)
        different_tasks_selected = c_random.sample(self.get_tasks_ids(), 3)

        problems_selected_str = [
            (different_image_ids_selected[0], None),
            (different_image_ids_selected[1], None),
            (different_image_ids_selected[2], different_tasks_selected[0]),
            (different_image_ids_selected[0], different_tasks_selected[1]),
            (different_image_ids_selected[3], different_tasks_selected[2]),
            (different_image_ids_selected[1], different_tasks_selected[1])
        ]
        num_tr = [
            None,
            None,
            self.num_tr_dp_lower_perf,
            self.num_tr_dp_lower_perf,
            self.num_tr_dp_lower_perf,
            self.num_tr_dp_lower_perf_few
        ]

        # num_tr = [self.num_tr_dp_max_perf] + [self.num_tr_dp_lower_perf] * (length - 1)

        problem_seq = BELLProblemSequence(problems_selected_str,
                                               self.batch_size, None,
                                               c_random_seed,
                                               num_tr,
                                               self.num_val,
                                               self.num_test)
        return problem_seq

    def get_S_out(self, c_random_seed: int, length=6):
        # S_1^{100%}, S_2^{60%}, S_3^{60%}, S_4^{60%}, S_5^{60%}, S_6=(D_1, T_6)^{60%}
        # Evaluating low-level transfer
        c_random = random.Random(c_random_seed)

        different_image_ids_selected = c_random.sample(self.get_all_image_dataset_ids(), length - 1)
        different_image_ids_selected += [different_image_ids_selected[0]]
        different_tasks_selected = c_random.sample(self.get_tasks_ids(), length)

        problems_selected_str = [(different_image_ids_selected[i], different_tasks_selected[i]) for i in range(length)]

        num_tr = [self.num_tr_dp_max_perf] + [self.num_tr_dp_lower_perf] * (length - 1)

        problem_seq = BELLProblemSequence(problems_selected_str,
                                               self.batch_size, None,
                                               c_random_seed,
                                               num_tr,
                                               self.num_val,
                                               self.num_test)
        return problem_seq

    def get_S_out_star(self, c_random_seed: int, length=6):
        # S_1^{60%}, S_2=(D_1, T_2)^{100%}, S_3^{60%}, S_4^{60%}, S_5^{60%}, S_6=(D_1, T_6)^{60%}
        # Evaluating low-level transfer
        c_random = random.Random(c_random_seed)

        different_image_ids_selected = c_random.sample(self.get_all_image_dataset_ids(), length - 2)
        different_image_ids_selected = [different_image_ids_selected[0]] + different_image_ids_selected + [
            different_image_ids_selected[0]]
        different_tasks_selected = c_random.sample(self.get_tasks_ids(), length)

        problems_selected_str = [(different_image_ids_selected[i], different_tasks_selected[i]) for i in range(length)]

        num_tr = [self.num_tr_dp_lower_perf] + [self.num_tr_dp_max_perf] + [self.num_tr_dp_lower_perf] * (length - 2)

        problem_seq = BELLProblemSequence(problems_selected_str,
                                               self.batch_size, None,
                                               c_random_seed,
                                               num_tr,
                                               self.num_val,
                                               self.num_test)
        return problem_seq

    def get_S_out_star_star(self, c_random_seed: int, length=6):
        # S_1^{60%}, S_2=(D_1, T_2)^{100%}, S_3^{60%}, S_4^{60%}, S_5^{60%}, S_1^{60%}
        # Evaluating low-level transfer
        c_random = random.Random(c_random_seed)

        different_image_ids_selected = c_random.sample(self.get_all_image_dataset_ids(), length - 2)
        different_task_ids_selected = c_random.sample(self.get_tasks_ids(), length - 1)

        p1_p2_shared_img_domain_id = different_image_ids_selected.pop(0)
        p1_task_id = different_task_ids_selected.pop(0)

        p1 = (p1_p2_shared_img_domain_id, p1_task_id)
        p2 = (p1_p2_shared_img_domain_id, different_task_ids_selected.pop(0))

        problems_selected_str = [p1, p2]
        problems_selected_str += [(different_image_ids_selected[j], different_task_ids_selected[j]) \
                                  for j in range(length - 3)]

        problems_selected_str.append(p1)

        num_tr = [self.num_tr_dp_lower_perf, self.num_tr_dp_max_perf]
        num_tr += [self.num_tr_dp_lower_perf] * (length - 3)
        num_tr += [self.num_tr_dp_lower_perf]

        problem_seq = BELLProblemSequence(problems_selected_str,
                                               self.batch_size, None,
                                               c_random_seed,
                                               num_tr,
                                               self.num_val,
                                               self.num_test)

        return problem_seq

