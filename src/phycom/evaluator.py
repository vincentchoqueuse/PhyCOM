import numpy as np
from ..core import SERLoss, MSELoss


class Evaluator():

    def __init__(self, model, alphabet, reduction="mean"):
        self.model = model
        self.alphabet = alphabet
        self.mse_criterion = MSELoss(name="mse_loss")
        self.ser_criterion = SERLoss(alphabet, demodulate=True, name="ser_loss")
        self.reduction = reduction

    def eval(self, dataset):

        constellation_list = []
        mse_values = []
        ser_values = []

        self.model.eval()

        for indice in range(len(dataset)):

            y, x_target = dataset[indice]
            x_pred = self.model(y)

            # compute criterion
            mse_value = self.mse_criterion(x_pred, self.alphabet[x_target])
            ser_value = self.ser_criterion(x_pred, x_target)
            mse_values.append(mse_value)
            ser_values.append(ser_value)

            constellation_list.append(x_pred)

        if self.reduction == "mean":
            mse = np.mean(np.array(mse_values))
            ser = np.mean(np.array(ser_values))

        logs = {"criterion": {"mse": mse, "ser": ser},
                "constellation": np.hstack(constellation_list)}

        return logs
