import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import copy

sys.path.insert(0, '../src')

from loss import MSELoss
from dsp import get_alphabet
from dataset import Chain1Dataset
from module import Demux
from phycom.model import PhyCOM
from phycom.module import IQ_imbalance, CFO, Inverse_FIR, IQ_imbalance, Demodulator
from phycom.trainer import Supervised_Trainer


class Custom_Supervised_Trainer(Supervised_Trainer):

    def __init__(self, model, kwargs=None):
        self.model = model
        self.selector = Demux()
        self._saved_model = []

    def f(self, params):
        error_real = super().f(params)
        model_temp = copy.deepcopy(self.model)
        self._saved_model.append(model_temp)  # we save the models for each iteration
        return error_real

    def train(self, x, y_set, verbose=0):
        super().train(x, y_set)
        return self._saved_model


def save_constellation(model, y, folder, filename):
    model(y)
    x_pred_pre_detector = model.get_data("detector")
    filename = "{}/constellation_{}".format(folder,filename)
    X = np.transpose(np.vstack([np.real(x_pred_pre_detector), np.imag(x_pred_pre_detector)]))
    status = pd.DataFrame(X, columns=["real", "imag"]).to_csv(filename)
    return status


if __name__ == '__main__':

    # parameter of the simulation
    N = 50
    N_tot = 500
    SNR = 30
    alphabet = get_alphabet("QAM", 16)
    dataset = Chain1Dataset("data/test.csv", alphabet, N_tot, 1, snr_dB=SNR)
    folder = "../results/simulation1"

    # single test
    y, x_target = dataset[0]

    # ------------------------------#
    #       Phycom detector         #
    # ------------------------------#
    model = PhyCOM(OrderedDict([
        ('iq2', IQ_imbalance()),
        ('cfo', CFO()),
        ('fir', Inverse_FIR([1, 0, 0, 0, 0, 0, 0, 0])),
        ('iq1', IQ_imbalance()),
        ('detector', Demodulator(alphabet))
        ]))

    # save initial data output
    save_constellation(model, y, folder, "before.csv")
    saved_model_tot = [[dataset.get_clairvoyant()]]

    # training
    N_vect = np.arange(N)
    demux = Demux(N_vect)
    demux.set_output(0)  # set training data
    x_set = (demux(N_vect), demux(x_target))

    trainer = Custom_Supervised_Trainer(model)
    saved_models = trainer.train(y, x_set)
    saved_model_tot.append(saved_models)
    save_constellation(saved_models[-1], y, folder, "after_training_step1.csv")

    # second pass (self labelling)
    N_vect = np.arange(N_tot)
    x_target_est = model(y)
    x_set = (N_vect, x_target_est)

    trainer = Custom_Supervised_Trainer(model)
    saved_models = trainer.train(y, x_set)
    saved_model_tot.append(saved_models)
    save_constellation(saved_models[-1], y, folder, "after_training_step2.csv")

    # evaluation (we compute the MSE for all the saved network)
    name = ["clairvoyant", "first_pass", "second_pass"]
    mse_criterion = MSELoss()

    # The list saved_models contains the
    # - clairvoyant model (perfect knowledge of the impairments)
    # - a list of models obtained during the first pass (one model by iteration)
    # - a list of models obtained during the second pass (one model by iteration)

    for index1 in range(len(name)):  #loop over the three techniques
        saved_models = saved_model_tot[index1]  #get the list of models
        data = []

        for index2 in range(len(saved_models)):  #we loop for each iteration

            model_temp = saved_models[index2]
            model_temp(y)
            x_pred_pre_detector = model_temp.get_data("detector")
            data_temp = []

            # save constellation
            if ((index1 == 1) and (index2 < 6)):  #we save the constellation for the first 6 iterations of the first pass (see figure 5 of paper)
                save_constellation(model_temp, y, folder, "after_training_step1_{}.csv".format(index2))

            # evaluate MSE for training dataset
            demux.set_output(0)  # set training dataset
            x_target_temp = demux(x_target)
            x_pred_pre_detector_temp = demux(x_pred_pre_detector)
            mse_value = mse_criterion(x_pred_pre_detector_temp, alphabet[x_target_temp])
            data_temp.append(mse_value)

            # evaluate MSE for testing dataset
            demux.set_output(1)  # set testing dataset
            x_target_temp = demux(x_target)
            x_pred_pre_detector_temp = demux(x_pred_pre_detector)
            mse_value = mse_criterion(x_pred_pre_detector_temp, alphabet[x_target_temp])
            data_temp.append(mse_value)

            # save data
            data.append(data_temp)

        filename = "{}/mse_{}.csv".format(folder,name[index1])
        df = pd.DataFrame(data, columns=["train", "test"])
        df.to_csv(filename)
        
        if index1 > 0:
            df.plot(logy=True, ylim=[0.001,2], grid=True, title = name[index1], xlabel="num iterations", ylabel="MSE")

    print("file saved in the folder results/simulation1")
    plt.show()