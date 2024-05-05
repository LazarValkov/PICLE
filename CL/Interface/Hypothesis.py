from abc import ABC, abstractmethod


class Hypothesis(ABC):
    @abstractmethod
    def __call__(self, inputs):
        pass

    @abstractmethod
    def save_to_a_single_file(self, filepath: str):
        pass
