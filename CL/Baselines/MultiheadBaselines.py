import time
import torch
from torch.nn import functional as F
from torch import autograd
from typing import Tuple, Union, List, Optional, Generator, Dict
import numpy as np
from torch.utils.data import DataLoader

from CL.Interface.CLAlgorithm import CLAlg
from CL.Interface.Hypothesis import Hypothesis
from CL.Interface.ProblemSolvingLog import ProblemSolvingLog
from CL.Interface.Problem import Problem
from CL.Interface.NNEvaluator import NNEvaluator
from CL.Interface.NNTrainer import NNTrainer
from CL.Interface.ModularCL.Library import Library
from CL.Interface.NNModule import NNModule
from CL.Interface.NNArchitecture import NNArchitecture


class Finetune(CLAlg):
    # Adapt the multi-head baseline for our CL setting (allowing for diff architecture to be used)
    # simplest type: a finetuning algorithm which can operate across diff architectures.
    def __init__(self, random_seed: int, device: str, num_epochs: int,
                 lib_folder: str
                 ):
        super().__init__(random_seed, device, num_epochs)

        self.enable_finetuning = True
        self.if_finetuning_finetune_a_copy = False

        # stores all the modules trained so far
        self.lib = Library(device)
        self.lib_folder = lib_folder
        self.output_heads_names = []

    def get_name(self):
        return "finetunig"

    def get_new_model_module_strs(self, problem):
        # try to overlap as many parameters as possible (but don't re-use output heads)
        c_module_types = problem.architecture_class.get_module_types()
        c_model_modules_str = []

        # for each type
        for c_module_type in c_module_types:
            c_lib_item_found = False
            if c_module_type in self.lib.items_by_module_type.keys():
                ## get all items for module type
                c_lib_items = self.lib.items_by_module_type[c_module_type]
                ### for each existing library item of this type
                ### if it's in output_heads, pass
                ### (find the first L )
                for li in c_lib_items:
                    if li.name not in self.output_heads_names:
                        c_lib_item_found = True
                        c_model_modules_str.append(li.name)
                        break

            if not c_lib_item_found:
                c_model_modules_str.append(None)

        return tuple(c_model_modules_str)

    def _get_model_from_program_strs(self,
                                     program_strs: Tuple[Union[str, None]],
                                     architecture_class,
                                     output_dim,
                                     random_init_random_seed=None
                                     ) -> NNArchitecture:
        modules = []
        for m_str in program_strs:
            modules.append(None if m_str is None else self.lib[m_str].module)
        modules = tuple(modules)

        model = architecture_class(modules, self.device,
                                   self.enable_finetuning,
                                   self.if_finetuning_finetune_a_copy,
                                   random_init_random_seed=random_init_random_seed,
                                   output_dim=output_dim)

        return model

    def solve_problem(self, problem: Problem, trainer: NNTrainer, max_time: float = None) \
            -> (Hypothesis, ProblemSolvingLog):
        c_random_seed = problem.num_id + self.random_seed if self.random_seed is not None else None

        assert trainer is not None

        c_model_modules_str = self.get_new_model_module_strs(problem)
        c_model = self._get_model_from_program_strs(c_model_modules_str, problem.architecture_class,
                                                    problem.output_dim, c_random_seed)

        start_time = time.time()
        trainer.train(c_model, self.num_epochs)
        training_time = time.time() - start_time

        self._update_state_and_save_lib(problem, c_model_modules_str, c_model)
        return self._perform_last_evaluation_and_return_results(problem, c_model, training_time)

    def _perform_last_evaluation_and_return_results(self, problem, c_model, training_time):
        evaluator = NNEvaluator(problem, self.device)
        val_loss, val_acc = evaluator.evaluate_on_val_dataset(c_model)
        test_loss, test_acc = evaluator.evaluate_on_test_dataset(c_model)

        memory_used = torch.cuda.memory_allocated(device=self.device)

        problem_solving_log = ProblemSolvingLog(self.random_seed,
                                                is_alg_modular=False,
                                                time_taken=training_time,
                                                memory_used=memory_used,
                                                val_loss=val_loss, val_acc=val_acc,
                                                test_loss=test_loss, test_acc=test_acc)

        return c_model, problem_solving_log

    def _update_state_and_save_lib(self, problem, c_model_modules_str, c_model):
        c_model_modules_str_updated = []
        # update library
        for i, c_module_str in enumerate(c_model_modules_str):
            if c_module_str is None:
                c_module = c_model.modules[i]
                c_module_str_new = f"{problem.num_id}_{c_module.get_module_type_name()}"
                self.lib.add_module(c_module_str_new, c_module, None, None, -1)

                c_model_modules_str_updated.append(c_module_str_new)
            else:
                c_model_modules_str_updated.append(c_module_str)

        # update "output heads"
        c_output_head_str = c_model_modules_str_updated[-1]
        self.output_heads_names.append(c_output_head_str)

        # update "previous solutions"
        self.lib.previous_solutions.append(c_model_modules_str_updated)

        # save the library
        assert problem.num_id is not None
        assert self.lib_folder is not None
        new_lib_folder = f"{self.lib_folder}/{problem.num_id}/"
        self.lib.save(new_lib_folder, problem.num_id, save_all_modules=True)

    def get_hypothesis_from_a_single_file(self, filepath: str, problem: Problem) -> Hypothesis:
        num_modules = problem.architecture_class.get_num_modules()
        standalone_prog = tuple(None for _ in range(num_modules))
        hypothesis = problem.architecture_class(standalone_prog,
                                                self.device, self.enable_finetuning, self.if_finetuning_finetune_a_copy,
                                                output_dim=problem.output_dim)
        hypothesis.load_modules_from_a_single_file(filepath, self.device)
        return hypothesis


class EWC(Finetune):
    def __init__(self, random_seed: int, device: str, num_epochs: int,
                 lib_folder: str, fisher_lambda=1000, fisher_sample_size=256):
        super().__init__(random_seed, device, num_epochs, lib_folder)

        # using default parameters
        self.fisher_lambda = fisher_lambda
        self.fisher_sample_size = fisher_sample_size
        self.additional_info_dict = {}

    def get_name(self):
        return f"ewc_{self.fisher_lambda}"

    def __estimate_fisher_diagonal(self, c_model, train_loader, is_multiple_output):
        log_likelihoods = []
        samples_so_far = 0
        # train_loader, _ = self.benchmark.load(self.current_task, batch_size=32)
        for x, y in train_loader:
            batch_size = len(y)
            x = x.to(self.device)
            y = y.to(self.device)

            c_logit, c_out = c_model(x)
            if is_multiple_output:
                log_out = F.log_softmax(c_logit, dim=1)
                log_likelihoods.append(log_out[range(batch_size), y.data])
            else:
                log_out = F.binary_cross_entropy_with_logits(c_logit, y, reduction='none')
                log_likelihoods.append(log_out)

            samples_so_far += batch_size
            if samples_so_far > self.fisher_sample_size:
                break

        log_likelihoods = torch.cat(log_likelihoods).unbind()
        grads = zip(*[autograd.grad(l, c_model.get_trainable_parameters_apart_last_module(),
                                    retain_graph=(i < len(log_likelihoods))) \
                      for i, l in enumerate(log_likelihoods, 1)])
        grads = [torch.stack(grad) for grad in grads]
        fisher_diagonals = [(grad ** 2).mean(0) for grad in grads]

        # note for the next line: in pytorch, module names are like W1.weight
        # but, we can't get attrs using getattr('W1.weight') because of the nested call (dot)
        # one trick is to replace the '.' with '_'
        # the other tick is to use: functools.reduce(getattr, [obj] + attr.split('.'))
        param_names = [n.replace('.', '_') for n, p in c_model.get_trainable_named_parameters_apart_last_module()]
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def __consolidate(self, c_model, train_loader, is_multiple_output):
        fisher_diagonals = self.__estimate_fisher_diagonal(c_model, train_loader, is_multiple_output)
        for name, param in c_model.get_trainable_named_parameters_apart_last_module():
            name = name.replace('.', '_')
            self.additional_info_dict[f"{name}_mean"] = param.data.clone()
            self.additional_info_dict[f"{name}_fisher"] = fisher_diagonals[name].data.clone()

    def __calculate_ewc_loss(self, c_model):
        # shouldn't be called for the first task
        # because we have not consolidated anything yet
        losses = []

        for name, param in c_model.get_trainable_named_parameters_apart_last_module():
            name = name.replace('.', '_')
            if f"{name}_mean" not in self.additional_info_dict.keys():
                # regularise only the previously trained modules
                continue

            mean = self.additional_info_dict[f"{name}_mean"]
            fisher = self.additional_info_dict[f"{name}_fisher"]
            losses.append((fisher * (param - mean) ** 2).sum())

        return (self.fisher_lambda / 2.0) * sum(losses)

    def solve_problem(self, problem: Problem, trainer: NNTrainer, max_time: float = None) \
            -> (Hypothesis, ProblemSolvingLog):
        c_random_seed = problem.num_id + self.random_seed if self.random_seed is not None else None

        assert trainer is not None

        c_model_modules_str = self.get_new_model_module_strs(problem)
        c_model = self._get_model_from_program_strs(c_model_modules_str, problem.architecture_class,
                                                    problem.output_dim, c_random_seed)
        start_time = time.time()

        if problem.num_id == 0:
            trainer.train(c_model, self.num_epochs)
        else:
            ewc_extra_loss = lambda: self.__calculate_ewc_loss(c_model)
            trainer.train(c_model, self.num_epochs, additional_loss_fn=ewc_extra_loss)

        training_time = time.time() - start_time

        is_multiple_output = problem.architecture_class.get_criterion_class() == torch.nn.CrossEntropyLoss
        self.__consolidate(c_model, problem.get_tr_data_loader(), is_multiple_output)

        self._update_state_and_save_lib(problem, c_model_modules_str, c_model)
        return self._perform_last_evaluation_and_return_results(problem, c_model, training_time)


class PerpetualDataLoader():
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iterator = iter(self.data_loader)

    def get_next_batch(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.data_loader)
            batch = next(self.iterator)

        return batch


class ERwRingBuffer(Finetune):
    """ implementing "On Tiny Episodic Memories in Continual Learning", https://arxiv.org/abs/1902.10486 """

    def __init__(self, random_seed: int, device: str, num_epochs: int,
                 lib_folder: str, num_examples_to_store_per_class):
        super().__init__(random_seed, device, num_epochs, lib_folder)

        self.num_examples_to_store_per_class = num_examples_to_store_per_class
        self.memory = []  # List<TensorDataSet>
        self.memory_data_loaders = []
        self.previous_solutions = []
        self.previous_criterions = []

    def get_name(self):
        return f"ERRing_{self.num_examples_to_store_per_class}"

    def update_memory(self, problem):
        # Note: Assuming only classification tasks.
        num_classes = problem.output_dim if problem.output_dim > 1 else 2

        # num_data_points_per_class = self.num_examples_to_store_per_problem // num_classes
        num_data_points_per_class = self.num_examples_to_store_per_class

        tr_data_loader = problem.get_tr_data_loader()
        c_dataset = tr_data_loader.dataset
        all_inputs, all_labels = c_dataset.tensors[0], c_dataset.tensors[1]
        if hasattr(c_dataset, 'transform') and c_dataset.transform is not None:
            all_inputs = c_dataset.transform(all_inputs)
        all_labels_flattened = torch.flatten(all_labels)

        memory_inputs_list, memory_labels_list = [], []
        for c_class in range(num_classes):
            relevant_indices = (all_labels_flattened == c_class).nonzero(as_tuple=True)[0]
            # truncate relevant_indices to the last num_data_points_per_class examples
            relevant_indices = relevant_indices[-num_data_points_per_class:]
            relevant_inputs = all_inputs[relevant_indices]
            relevant_outputs = all_labels[relevant_indices]

            memory_inputs_list.append(relevant_inputs)
            memory_labels_list.append(relevant_outputs)

        memory_inputs = torch.cat(memory_inputs_list)
        memory_labels = torch.cat(memory_labels_list)

        c_tds = torch.utils.data.TensorDataset(memory_inputs, memory_labels)
        self.memory.append(c_tds)

    def __calculate_er_loss(self):
        all_losses_sum = 0.
        for i, prev_solution in enumerate(self.previous_solutions):
            # sample batch
            inputs, labels = self.memory_data_loaders[i].get_next_batch()
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # forward the pass
            logits, output = prev_solution(inputs)

            # compute the loss
            c_loss = self.previous_criterions[i](logits, labels)

            # add to the sum of losses
            all_losses_sum += c_loss

        return all_losses_sum

    def solve_problem(self, problem: Problem, trainer: NNTrainer, max_time: float = None) \
            -> (Hypothesis, ProblemSolvingLog):
        c_random_seed = problem.num_id + self.random_seed if self.random_seed is not None else None

        assert trainer is not None

        # populate memory data loaders
        prev_batch_size = problem.batch_size // len(self.previous_solutions) if len(self.previous_solutions) > 0 else -1
        self.memory_data_loaders = [PerpetualDataLoader(DataLoader(ds, prev_batch_size, shuffle=True)) for ds in
                                    self.memory]

        c_model_modules_str = self.get_new_model_module_strs(problem)
        c_model = self._get_model_from_program_strs(c_model_modules_str, problem.architecture_class,
                                                    problem.output_dim, c_random_seed)
        start_time = time.time()

        if problem.num_id == 0:
            trainer.train(c_model, self.num_epochs)  # add the function of additional loss to the arguments here.
        else:
            trainer.train(c_model, self.num_epochs, additional_loss_fn=self.__calculate_er_loss)
        training_time = time.time() - start_time

        # reset memory data loaders
        self.memory_data_loaders = []

        self.update_memory(problem)
        self.previous_solutions.append(c_model)
        self.previous_criterions.append(problem.architecture_class.get_criterion_class()(reduction="mean"))

        self._update_state_and_save_lib(problem, c_model_modules_str, c_model)
        return self._perform_last_evaluation_and_return_results(problem, c_model, training_time)

