import torch
import torch.nn as nn
from CL.Interface.NNModule import NNModule
import torch.nn.functional as F
from Experiments.BELL.DataCreation.ImageDatasetProviders import ImageDatasetProvider

T1_FC_NUM_HIDDEN_UNITS = 64
T1_MODULES_LAST_HIDDEN_STATE_NUM_UNITS = T1_FC_NUM_HIDDEN_UNITS
T1_MODULES_NUM_HIDDEN_UNITS = 64
T1_MODULES_CNN_IN_CH = 1
T1_MODULES_CNN_SM_OUT_DIM = ImageDatasetProvider.NUM_CLASSES_PER_PROBLEM
T1_MODULES_CNN_LIN_OUT_DIM = 8
T1_MODULES_LIST_LENGTH = 2  # 10 # 9
T1_MODULES_FC_IN_DIM = T1_MODULES_LIST_LENGTH * T1_MODULES_CNN_LIN_OUT_DIM  # num_output_class * list length
T1_MODULES_FC_NUM_LAYERS = 3

ACTIVATION_FUNCTION = F.relu


class T1NNModule(NNModule):
    @staticmethod
    def _initialise_weights(m: nn.Module, nonlinearity: str):
        # Linear / Conv2D / Sigmoid /
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(nonlinearity))
        m.bias.data.fill_(0.)



# T1Module_CNN_ConvL1, T1Module_CNN_ConvL2, T1Module_CNN_FCL1, T1Module_CNN_FCL2, T1Module_CNN_FCL3
# T1Module_MLP_FCL1, T1Module_MLP_FCL2, T1Module_MLP_FCL3
# T2Module_MLP_FCL1, T2Module_MLP_FCL2, T2Module_MLP_FCL3, T2Module_MLP_FCL4


class T1Module_CNN_ConvL1(T1NNModule):
    @staticmethod
    def get_module_type_name():
        return "L1"

    def __init__(self):
        super(T1Module_CNN_ConvL1, self).__init__()
        self.cl1 = nn.Conv2d(T1_MODULES_CNN_IN_CH, T1_MODULES_NUM_HIDDEN_UNITS, kernel_size=5, stride=2, padding=0)
        self._initialise_weights(self.cl1, 'relu')

    def _forward(self, x, return_logits_as_well=False):
        logits = self.cl1(x)
        h1 = ACTIVATION_FUNCTION(logits)
        return (logits, h1) if return_logits_as_well else h1


class T1Module_CNN_ConvL2(T1NNModule):
    @staticmethod
    def get_module_type_name():
        return "L2"

    def __init__(self):
        super(T1Module_CNN_ConvL2, self).__init__()
        self.cl2 = nn.Conv2d(T1_MODULES_NUM_HIDDEN_UNITS, T1_MODULES_NUM_HIDDEN_UNITS, kernel_size=5, stride=2, padding=0)
        self._initialise_weights(self.cl2, 'relu')

    def _forward(self, x, return_logits_as_well=False):
        logits = self.cl2(x)
        h2 = ACTIVATION_FUNCTION(logits)
        return (logits, h2) if return_logits_as_well else h2


class T1Module_CNN_FCL1(T1NNModule):
    @staticmethod
    def get_module_type_name():
        return "L3"

    def __init__(self):
        super(T1Module_CNN_FCL1, self).__init__()
        self.fc1 = nn.Linear(4 * 4 * T1_MODULES_NUM_HIDDEN_UNITS, T1_MODULES_NUM_HIDDEN_UNITS)
        self._initialise_weights(self.fc1, 'relu')

    def _forward(self, x, return_logits_as_well=False):
        logits = self.fc1(x)
        h4 = ACTIVATION_FUNCTION(logits)
        return (logits, h4) if return_logits_as_well else h4


class T1Module_CNN_FCL2(T1NNModule):
    @staticmethod
    def get_module_type_name():
        return "L4"

    def __init__(self):
        super(T1Module_CNN_FCL2, self).__init__()
        self.fc2 = nn.Linear(T1_MODULES_NUM_HIDDEN_UNITS, T1_MODULES_NUM_HIDDEN_UNITS)
        self._initialise_weights(self.fc2, 'relu')

    def _forward(self, x, return_logits_as_well=False):
        logits = self.fc2(x)
        h4 = ACTIVATION_FUNCTION(logits)
        return (logits, h4) if return_logits_as_well else h4


class T1Module_CNN_FCL3(T1NNModule):
    @staticmethod
    def get_module_type_name():
        return "L5_L"

    def __init__(self):
        super(T1Module_CNN_FCL3, self).__init__()
        self.fc3 = nn.Linear(T1_MODULES_NUM_HIDDEN_UNITS, T1_MODULES_CNN_LIN_OUT_DIM)
        self._initialise_weights(self.fc3, 'linear')

    def _forward(self, x, return_logits_as_well=False):
        logits = self.fc3(x)
        # outputs = logits
        outputs = torch.softmax(logits, dim=1)
        return (logits, outputs) if return_logits_as_well else outputs


class T1Module_MLP_FCL1(T1NNModule):
    @staticmethod
    def get_module_type_name():
        return "L6"

    def __init__(self):
        super(T1Module_MLP_FCL1, self).__init__()
        self.fc4 = nn.Linear(T1_MODULES_FC_IN_DIM, T1_FC_NUM_HIDDEN_UNITS)
        self._initialise_weights(self.fc4, 'relu')

    def _forward(self, x, return_logits_as_well=False):
        logits = self.fc4(x)
        outputs = ACTIVATION_FUNCTION(logits)
        return (logits, outputs) if return_logits_as_well else outputs


class T1Module_MLP_FCL2(T1NNModule):
    @staticmethod
    def get_module_type_name():
        return "L7"

    def __init__(self):
        super(T1Module_MLP_FCL2, self).__init__()
        self.fc5 = nn.Linear(T1_FC_NUM_HIDDEN_UNITS, T1_FC_NUM_HIDDEN_UNITS)
        self._initialise_weights(self.fc5, 'relu')

    def _forward(self, x, return_logits_as_well=False):
        logits = self.fc5(x)
        outputs = ACTIVATION_FUNCTION(logits)
        return (logits, outputs) if return_logits_as_well else outputs


class T1Module_MLP_FCL3(T1NNModule):
    @staticmethod
    def get_module_type_name():
        return "L8"

    def __init__(self):
        super(T1Module_MLP_FCL3, self).__init__()
        self.fc6 = nn.Linear(T1_FC_NUM_HIDDEN_UNITS, 1)
        self._initialise_weights(self.fc6, 'sigmoid')

    def _forward(self, x, return_logits_as_well=False):
        logits = self.fc6(x)
        outputs = torch.sigmoid(logits)
        return (logits, outputs) if return_logits_as_well else outputs


class T2Module_MLP_FCL1(T1NNModule):
    @staticmethod
    def get_module_type_name():
        return "T2L1"

    def __init__(self):
        super(T2Module_MLP_FCL1, self).__init__()
        # self.fc1 = nn.Linear(10, T1_FC_NUM_HIDDEN_UNITS)
        self.fc1 = nn.Linear(784, T1_FC_NUM_HIDDEN_UNITS)
        self._initialise_weights(self.fc1, 'relu')

    def _forward(self, x, return_logits_as_well=False):
        logits = self.fc1(x)
        outputs = ACTIVATION_FUNCTION(logits)
        return (logits, outputs) if return_logits_as_well else outputs


class T2Module_MLP_FCL2(T1NNModule):
    @staticmethod
    def get_module_type_name():
        return "T2L2"

    def __init__(self):
        super(T2Module_MLP_FCL2, self).__init__()
        self.fc2 = nn.Linear(T1_FC_NUM_HIDDEN_UNITS, T1_FC_NUM_HIDDEN_UNITS)
        self._initialise_weights(self.fc2, 'relu')

    def _forward(self, x, return_logits_as_well=False):
        logits = self.fc2(x)
        outputs = ACTIVATION_FUNCTION(logits)
        return (logits, outputs) if return_logits_as_well else outputs


class T2Module_MLP_FCL3(T1NNModule):
    @staticmethod
    def get_module_type_name():
        return "T2L3"

    def __init__(self):
        super(T2Module_MLP_FCL3, self).__init__()
        self.fc3 = nn.Linear(T1_FC_NUM_HIDDEN_UNITS, T1_FC_NUM_HIDDEN_UNITS)
        self._initialise_weights(self.fc3, 'relu')

    def _forward(self, x, return_logits_as_well=False):
        logits = self.fc3(x)
        outputs = ACTIVATION_FUNCTION(logits)
        return (logits, outputs) if return_logits_as_well else outputs


class T2Module_MLP_FCL4(T1NNModule):
    @staticmethod
    def get_module_type_name():
        return "T2L4"

    def __init__(self):
        super(T2Module_MLP_FCL4, self).__init__()
        self.fc4 = nn.Linear(T1_MODULES_NUM_HIDDEN_UNITS, T1_MODULES_CNN_LIN_OUT_DIM)
        self._initialise_weights(self.fc4, 'linear')

    def _forward(self, x, return_logits_as_well=False):
        logits = self.fc4(x)
        outputs = torch.softmax(logits, dim=1)
        return (logits, outputs) if return_logits_as_well else outputs
