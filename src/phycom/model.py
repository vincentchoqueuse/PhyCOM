from collections import OrderedDict
from typing import Any
from model import Sequential
import numpy as np


class PhyCOM(Sequential):

    r"""

    .. code ::

        model = PhyCOM(
                IQ_imbalance([1,0.1,-0.2,0.9]),
                CFO(0.1),
                IQ_imbalance([1,0.1,-0.1,0.5]),
                )
    """

    def __init__(self, *args: Any):

        self._modules = OrderedDict()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    @property
    def grad(self):
        grad = []
        for _, module in self._modules.items():
            if module.grad is not None:
                grad.append(module.grad)

        grad = np.hstack(grad)
        return grad

    def compute_H(self, N):

        I_mat = np.eye(2*N)
        H = I_mat
        for _, module in self._modules.items():
            H_temp = module.compute_H(N)
            H = np.matmul(H_temp, H)
        return H

    def zero_grad(self):
        for _, module in self._modules.items():
            module.zero_grad()

    def train(self):
        for _, module in self._modules.items():
            module.train()

    def eval(self):
        for _, module in self._modules.items():
            module.eval()

    def forward(self, y):

        for _, module in self._modules.items():
            y = module(y)

        return y

    def backward(self, B):

        modules_reverse = OrderedDict(reversed(list(self._modules.items())))

        for name, module in modules_reverse.items():
            B = module.backward(B)

        return B

    def __str__(self):
        str = "Phycom Network\n"
        for _, module in self._modules.items():
            str += "{}\n".format(module)

        return str
