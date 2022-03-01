import numpy as np
import matplotlib.pyplot as plt


class Module():

    r"""
     Module

    Select a subset of vector.

    Parameters
    ----------
    name : string, optional
        Layer Name

    """
    _bypass = False
    _parameters = []
    _frozen = False
    _training = False

    def __init__(self, parameters, name="layer"):
        self.name = name
        self._parameters = parameters

    def forward(self, x):
        return x

    def nb_parameters(self):
        return len(self._parameters)

    def set_parameter(self, parameter):
        self._parameters = [parameter]

    def set_parameters(self, parameters):
        self._parameters = parameters

    def get_parameters(self):
        return self._parameters

    def get_parameter(self):
        return self._parameters[0]

    def set_frozen(self, frozen):
        self._frozen = frozen

    def set_bypass(self, value):
        self._bypass = value

    def get_data(self):
        return getattr(self, '_x', None)

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def __call__(self, x):
        self._x = x
        if not self._bypass:
            y = self.forward(x)
        else:
            y = x
        return y

    def __str__(self):
        return "<Layer {} (bypass={}): {}>".format(self.name, self._bypass, self._parameters)


class Linear_Module(Module):

    def __init__(self, parameters, name="layer"):
        self.name = name
        self.set_parameters(parameters)

    def compute_H(self, N):
        r"""
        Compute the transfer matrix
        """

        H = np.zeros((2*N, 2*N))
        for indice in range(2*N):
            x = np.zeros(N, dtype=np.complex)
            if indice < N:
                x[indice] = 1
            else:
                x[indice-N] = 1j
            y = self(x)
            H[:, indice] = np.hstack([np.real(y), np.imag(y)])

        return H

    def forward(self, x):
        raise Exception("Forward method must be overriden")

    def backward(self, H_input):
        H_output = np.matmul(H_input, self._H)
        return H_output


class Recorder(Module):

    r"""
    Recorder Module

    Parameters
    ----------
    name : string, optional
        Layer Name
    """

    def __init__(self, name="recorder"):
        self.name = name

    def get_data(self, index=None):
        if index is None:
            x = self._x
        else:
            x = self._x[index]
        return x

    def forward(self, x):
        self._x = x
        return x


class Scope(Module):

    def __init__(self, type, name="scope"):
        self.name = name
        self.type = type

    def forward(self, x):

        if self.type == "scatter":
            plt.figure(self.name)
            plt.plot(np.real(x), np.imag(x), ".")
            plt.xlabel("real part")
            plt.ylabel("imag part")
            plt.axis('equal')

        return x


class Demux(Linear_Module):

    def __init__(self, index=None, output=0, name="scope"):
        self.name = name
        self._index = index
        self._output = output

    def set_output(self, output):
        self._output = output

    def set_index(self, index):
        self._index = index

    def compute_H(self, N):

        r"""
        This method computes the layer transfer matrix.

        """
        # construct P
        index = self._index

        if index is None:
            P = np.eye(N)
        else:
            Np = len(index)
            P = np.zeros((Np, N))

            for indice in range(Np):
                P[indice, index[indice]] = 1

        H = np.kron(np.eye(2), P)
        return H

    def forward(self, x):

        if self._output == 0:
            x = x[self._index]

        if self._output == 1:
            x = np.delete(x, self._index)

        return x
