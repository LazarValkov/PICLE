from CL.Interface.NNArchitecture import NNArchitecture
from typing import Union, Tuple, Dict, List


class EvaluatePathResult:
    def __init__(self,
                 program_str: str,
                 program_strs_tuple: Union[Tuple, Tuple[str]],
                 val_loss: float, val_acc: float,
                 exploration_time: float,
                 model: NNArchitecture,
                 learning_curves_dict: Dict[str, List[float]]):
        self.program_str = program_str
        self.program_strs_tuple = program_strs_tuple

        self.val_loss = val_loss
        self.val_acc = val_acc

        self.exploration_time = exploration_time
        self.model = model
        self.learning_curves_dict = learning_curves_dict
