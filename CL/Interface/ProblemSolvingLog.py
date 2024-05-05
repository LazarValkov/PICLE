class ProblemSolvingLog:
    """
    Contains the data logged during problem-solving by a CL algorithm
    """
    def __init__(self,
                 random_seed: int,
                 is_alg_modular: bool,
                 time_taken: float,
                 val_loss: float, val_acc: float,
                 test_loss: float, test_acc: float,
                 memory_used: float = -1):
        self.random_seed = random_seed
        self.is_alg_modular = is_alg_modular
        self.time_taken = time_taken
        self.memory_used = memory_used

        self.val_loss = val_loss
        self.val_acc = val_acc
        self.test_loss = test_loss
        self.test_acc = test_acc
