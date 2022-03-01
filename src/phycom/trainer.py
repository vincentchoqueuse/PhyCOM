from module import Demux
from scipy.optimize import least_squares
import numpy as np
import copy

class Supervised_Trainer(object):

    def __init__(self, model, kwargs=None):
        self.model = model
        self.selector = Demux()
        self.kwargs = kwargs

    def f(self, params):
        # Be careful here. scipy assumes a cost function of the form  0.5 * sum(epsilon**2)
        # The function f must corresponds to epsilon et the jacobian must be computed according
        # to these cost function (and not the classical MSE)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

        self.model.set_parameters(params)
        self.model(self._x)

        # compute error
        x_pre_detector = self.model.get_data("detector")
        y_est = self.selector(x_pre_detector)
        y_target = self._y
        y_est_real = np.hstack([np.real(y_est), np.imag(y_est)])
        y_real = np.hstack([np.real(y_target), np.imag(y_target)])
        error_real = y_real - y_est_real
        return error_real

    def jac(self, params):
        self.model.zero_grad()
        N = len(self._x)
        H = self.selector.compute_H(N)
        self.model.backward(-H)
        grad = self.model.grad
        return grad

    def train(self, x, y_set, verbose=0):

        data_aided_index = y_set[0]
        y = y_set[1]
        alphabet = self.model.get_module("detector").get_alphabet()

        self._x = x
        self.report = []
        self._y = alphabet[y]
        self.selector.set_index(data_aided_index)
        self.model.train()

        params = list(self.model.parameters())
        params_init = np.hstack(params)
        results = least_squares(self.f, params_init, jac=self.jac, max_nfev=100, verbose=verbose)
        return results


class Semi_Supervised_Trainer():

    r"""
    Semi_Supervised_Trainer()

    Train a phyCOM network with a semi supervised approach composed of two steps:

    * A supervised step where some pilot samples are provided to the trainer
    * An unsupervised step using the detected symbols (self labelling)

    Parameters
    ----------
    model : phycom
        The Phycom model to be trained
    unsupervised_step : Boolean
        Specify if the unsupervised step must be performed after the supervised step
    """

    def __init__(self, model, unsupervised_step=True, name="trainer", kwargs=None):

        """
        Initialize the Semi_Supervised_Trainer baseclass.

        """
        self.model = model
        self.unsupervised_step = unsupervised_step
        self.name = name
        self.kwargs = kwargs

    def get_trainer(self):
        return Supervised_Trainer(self.model)

    def train(self, x, y_set, verbose=0):
        """
        This method trains the phycom network using some pilot samples

        Parameters
        ----------
        x : numpy array
            The input of the network
        y_set: list
            A list (index,data) containing the index and the value of some pilot samples.

        Returns
        -------
        model2 :  phycom
            The phyCOM model obtained after the two learning passes (supervised + unsupverised)
        model 3 : phycom
            The phyCOM model obtained after the supervised step

        """

        time_data = []
        trainer = self.get_trainer()
        trainer.train(x, y_set, verbose=verbose)

        model1 = copy.deepcopy(self.model)
        if self.unsupervised_step:
            y = self.model(x)
            y_set = (np.arange(len(x)), y)
            trainer.train(x, y_set, verbose=verbose)
            model2 = copy.deepcopy(self.model)
        else:
            model2 = copy.deepcopy(model1)

        self.time_data = time_data
        return model2, model1
