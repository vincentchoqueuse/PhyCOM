from collections import OrderedDict
from typing import Any


class Sequential():
    r"""
    .. code ::

        import numpy as np

        alpha = np.exp(2j*np.pi*np.arange(4)/4)
        model = Sequential(
                Modulator(alphabet),
                IQ_imbalance([1,0.1,-0.2,0.9]),
                CFO(0.1),
                IQ_imbalance([1,0.1,-0.1,0.5]))
    """

    def __init__(self, *args: Any):

        self._modules = OrderedDict()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __len__(self) -> int:
        return len(self._modules)

    def add_module(self, name, module):
        r"""Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(name, str):
            raise TypeError("module name should be a string. Got {}".format(type(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))

        self._modules[name] = module

    def parameters(self):
        for name, param in self.named_parameters():
            yield param

    def named_parameters(self):
        for name, module in self._modules.items():
            yield name, module._parameters

    def set_parameters(self, params):
        index = 0
        for name, module in self._modules.items():

            nb_parameters = module.nb_parameters()
            if nb_parameters > 0:
                parameters = params[index:index+nb_parameters]
                module.set_parameters(parameters)
                index += nb_parameters

    def get_module(self, name):
        return self._modules[name]

    def get_data(self, name):
        module = self.get_module(name)
        return module.get_data()

    def set_frozen(self, frozen):
        for name, module in self._modules.items():
            module.set_frozen(frozen)

    def set_bypass(self, name, value):
        module = self.get_module(name)
        module.set_bypass(value)

    def forward(self, x):

        for name, module in self._modules.items():
            x = module(x)
        self.N = len(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def __str__(self):
        str = "Sequential Object\n"
        for name, module in self._modules.items():
            str += "{}\n".format(module)

        return str
