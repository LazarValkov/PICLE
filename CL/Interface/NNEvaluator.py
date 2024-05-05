import numpy as np
import torch
from CL.Interface.Problem import Problem


class NNEvaluator:
    """
    Used to evaluate a NN's performance on a problem
    """
    def __init__(self, problem: Problem, device: str):
        self.problem = problem
        self.device = device
        self.criterion_class = problem.architecture_class.get_criterion_class()

    def evaluate_performance(self, model, ds_loader, verbal=False):
        criterion = self.criterion_class(reduction="sum")
        is_multiple_output = self.criterion_class == torch.nn.CrossEntropyLoss

        loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in ds_loader:
                data, target = data.to(self.device), target.to(self.device)
                logits, output = model(data)

                loss += criterion(logits, target).detach().item()

                if is_multiple_output:
                    pred_np = output.cpu().detach().numpy().argmax(axis=1)
                    target_np = target.cpu().detach().numpy()
                else:
                    pred_np = output.cpu().detach().numpy().round().astype(int)
                    target_np = target.cpu().detach().numpy().round().astype(int)

                correct += np.equal(pred_np, target_np).sum()

        loss /= len(ds_loader.dataset)

        if verbal:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                loss, correct, len(ds_loader.dataset),
                100. * correct / len(ds_loader.dataset)))

        return loss, correct / len(ds_loader.dataset)

    def evaluate_on_tr_dataset(self, model, verbal=False):
        model.eval()
        tr_data_loader = self.problem.get_tr_data_loader()
        return self.evaluate_performance(model, tr_data_loader, verbal)

    def evaluate_on_val_dataset(self, model, verbal=False):
        model.eval()
        val_data_loader = self.problem.get_val_data_loader()
        return self.evaluate_performance(model, val_data_loader, verbal)

    def evaluate_on_test_dataset(self, model, verbal=False):
        model.eval()
        test_data_loader = self.problem.get_test_data_loader()
        return self.evaluate_performance(model, test_data_loader, verbal)
