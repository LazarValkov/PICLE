import time
import sys
import torch
import torch.optim as optim
import numpy as np
from CL.Interface.Problem import Problem
from CL.Interface.NNTrainer import NNTrainer, _clone_hidden_state
from CL.Interface.NNEvaluator import NNEvaluator
from CL.Interface.NNArchitecture import NNArchitecture


class BELLNNTrainer(NNTrainer):
    def __init__(self, problem: Problem, device: str,
                 learning_rate=0.00015632046,
                 weight_decay=0.9749541784205452,
                 iterations_patience=6000):
        super().__init__(problem, device, learning_rate, weight_decay)
        self.iterations_patience = iterations_patience

    def train(self, model: NNArchitecture, num_epochs: int=300, additional_loss_fn=None):
        """
        :param model: a torch.module which can be trained
        """
        tr_data_loader = self.problem.get_tr_data_loader()

        start_time = time.time()

        nn_evaluator = NNEvaluator(self.problem, device=self.device)
        # criterion = self.criterion_class(reduction="mean")
        criterion = self.criterion_class(reduction="none")
        is_multiple_output = self.criterion_class == torch.nn.CrossEntropyLoss
        trainable_parameters = model.get_trainable_parameters()

        optimizer = optim.AdamW(trainable_parameters,
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)

        param_updates_per_epoch = len(tr_data_loader)
        # num_its_since_last_improvement = 0

        c_param_update_index = 0.
        param_update_index_of_current_best_model = 0.

        best_accuracy = -sys.float_info.max
        best_loss = sys.float_info.max
        best_performance_new_fns_states = []  # a list of state_dicts for each new neural module

        tr_plot_x_axis, tr_losses, tr_accs = [], [], []
        val_plot_x_axis, val_losses, val_accs = [], [], []

        # store the initial parameters to ensure that we never go below random chance accuracy
        def evaluate_validation_performance_and_update_if_necessary():
            nonlocal best_accuracy, best_loss, best_performance_new_fns_states, param_update_index_of_current_best_model

            model.eval()
            c_val_loss, c_val_acc = nn_evaluator.evaluate_on_val_dataset(model)

            if c_val_loss < best_loss:
                best_loss = c_val_loss
                best_accuracy = c_val_acc
                best_performance_new_fns_states = [_clone_hidden_state(tm.state_dict()) for tm in model.trainable_modules]

                param_update_index_of_current_best_model = c_param_update_index

            val_plot_x_axis.append(c_param_update_index)
            val_losses.append(c_val_loss)
            val_accs.append(c_val_acc)

        evaluate_validation_performance_and_update_if_necessary()

        init_tr_loss, init_tr_acc = nn_evaluator.evaluate_on_tr_dataset(model)
        tr_plot_x_axis.append(c_param_update_index)
        tr_losses.append(init_tr_loss)
        tr_accs.append(init_tr_acc)

        num_tr_datapoints = len(tr_data_loader.dataset)

        for epoch in range(1, num_epochs + 1):
            model.train()
            # print(f"Starting epoch {epoch} / {num_epochs}")
            c_epoch_tr_loss_sum = 0.
            c_epoch_tr_correct_pred = 0.

            for batch_idx, (data, target) in enumerate(tr_data_loader):
                c_param_update_index += 1
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                logits, output = model(data)
                c_losses = criterion(logits, target)

                c_loss = c_losses.mean()

                if additional_loss_fn is not None:
                    # print("Adding more Loss")
                    c_loss += additional_loss_fn()

                c_loss.backward()
                optimizer.step()

                c_epoch_tr_loss_sum += c_losses.sum().item()
                # calculate accuracy:
                if is_multiple_output:
                    pred_np = output.cpu().detach().numpy().argmax(axis=1)
                    target_np = target.cpu().detach().numpy()
                else:
                    pred_np = output.cpu().detach().numpy().round().astype(int)  # .reshape((-1,))
                    target_np = target.cpu().detach().numpy().round().astype(int)  # .reshape((-1,))

                c_epoch_tr_correct_pred += np.equal(pred_np, target_np).sum()

            c_epoch_tr_loss = c_epoch_tr_loss_sum / num_tr_datapoints
            c_epoch_tr_acc = c_epoch_tr_correct_pred / num_tr_datapoints

            tr_plot_x_axis.append(c_param_update_index)
            tr_losses.append(c_epoch_tr_loss)
            tr_accs.append(c_epoch_tr_acc)

            # at the end of each epoch, update the best performance if necessary
            evaluate_validation_performance_and_update_if_necessary()

            # --- check for early stopping
            if c_param_update_index - param_update_index_of_current_best_model >= self.iterations_patience:
                break

        print(f"Training finished after {epoch} epochs")

        optimizer.zero_grad()
        model.eval()

        # restore the most successful checkpoint
        for tm_idx, tm in enumerate(model.trainable_modules):
            c_state_dict = best_performance_new_fns_states[tm_idx]
            tm.load_state_dict(c_state_dict)

        training_time = time.time() - start_time

        learning_curves_dict = {"tr_plot_x_axis": tr_plot_x_axis, "tr_losses": tr_losses, "tr_accs": tr_accs,
                                "val_plot_x_axis": val_plot_x_axis, "val_losses": val_losses, "val_accs": val_accs}

        return training_time, learning_curves_dict