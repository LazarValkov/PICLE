from CL.Interface.ProblemSequence import ProblemSequence
from Experiments.BELL.Problem import BELLProblem
from typing import List, Tuple, Union


class BELLProblemSequence(ProblemSequence):
    def __init__(self,
                 problems_str: List[Tuple[Union[None, str], Union[None, str]]],
                 batch_size: int,
                 dataset_folder,
                 random_seed: int,
                 total_num_tr_datapoints: Tuple[Union[None, int], Union[None, int], Union[None, int]] = None,
                 total_num_val_datapoints: Union[None, int] = None,
                 total_num_test_datapoints: Union[None, int] = None
                 ):
        """
        :param total_num_tr_datapoints:
            None = use all available for training
            int = use that much for training
            (A: int, B: int, C: int) = generated A items, using B different images, C different task
        """

        if type(total_num_tr_datapoints) == list:
            assert len(total_num_tr_datapoints) == len(problems_str)
        else:
            total_num_tr_datapoints = [total_num_tr_datapoints for _ in range(len(problems_str))]

        # using random_seed+p_id, so if we have the same problem multiple times, it doesn't have the exact same dataset
        problems = [BELLProblem(p_str, batch_size, dataset_folder, random_seed + p_id,
                                     total_num_tr_datapoints[p_id], total_num_val_datapoints, total_num_test_datapoints,
                                     num_id=p_id)
                    for p_id, p_str in enumerate(problems_str)]
        super().__init__(problems)
