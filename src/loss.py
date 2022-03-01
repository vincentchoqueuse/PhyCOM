import numpy as np


class Loss():

    def __init__(self, reduction="mean", name="loss"):
        self.reduction = reduction
        self.name = name

    def compute(self, input, target):
        pass

    def __call__(self, input, target):
        error = self.compute(input, target)

        if self.reduction is None:
            output = error

        if self.reduction == "max":
            output = np.max(error)

        if self.reduction == "sum":
            output = np.sum(error)

        if self.reduction == "mean":
            output = np.mean(error)

        return output


class MSELoss(Loss):

    """
    MSELoss()
    A class to compute MSE-type cost function

    Parameters
    ----------
    reduction (optional) : str
        The reduction type (mean,max,sum)

    name (optional) : str
        The object name.

    Example
    -------

    .. code-block:: Python

        loss = MSELoss(reduction="mean")
        value = loss(x,y)

    """

    def __init__(self, reduction="mean", name="mse_loss"):
        self.reduction = reduction
        self.name = name

    def compute(self, input, target):
        error = np.abs(input-target)**2
        return error


class SERLoss(Loss):

    """
    SERLoss()
    A class to compute SER-type cost function

    Parameters
    ----------
    reduction (optional) : str
        The reduction type (mean,max,sum,error_real)

    name (optional) : str
        The object name.

    Example
    -------

    .. code-block:: Python

        loss = SERLoss(reduction="mean")
        value = loss(x,y)
    """

    def __init__(self, reduction="mean", name="ser_loss"):
        self.reduction = reduction
        self.name = name

    def compute(self, input, target):
        input = np.ravel(input)
        target = np.ravel(target)
        error = 1*(np.ravel(np.abs(input-target)) > 0)
        return error
