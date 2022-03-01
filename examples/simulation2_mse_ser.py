import sys
import numpy as np
from collections import OrderedDict

sys.path.insert(0, '../src')

import dsp
import channel
from loss import SERLoss, MSELoss
from model import Sequential
from module import Demux
from utils import MC_Report_Writer, Data_Writer
from phycom.model import PhyCOM
from phycom.trainer import Semi_Supervised_Trainer
from phycom.module import IQ_imbalance, CFO, Inverse_FIR, IQ_imbalance, Demodulator
from dataset import Chain1Dataset

if __name__ == '__main__':

    # simulation parameters
    N = 50
    N_tot = 500
    SNR = 30
    alphabet = dsp.get_alphabet("QAM", 16)
    M_testing = 10  # increase size to improve the estimation of the MSE / SER metrics
    # please note that the error floor for the DSP2 and DSP4 techniques
    # can be observed using a value of M_testing >100
    folder = "../results/simulation2"

    # prepare MC simulations
    columns = ["theo", "clairvoyant", "old1", "old2", "old3", "old4", "phycom1", "phycom2"]
    nb_columns = len(columns)
    dataset = Chain1Dataset("data/test.csv", alphabet, N_tot, M_testing, snr_dB=SNR)
    Nb_trials = len(dataset)  # number of trials (is equal to M_testing)
    Np_vect = range(14, 84, 4)  # number of pilots

    mse_train_writer = Data_Writer(folder, "mse_train.csv", index=Np_vect, columns=columns)
    ser_train_writer = Data_Writer(folder, "ser_train.csv", index=Np_vect, columns=columns)
    mse_test_writer = Data_Writer(folder, "mse_test.csv", index=Np_vect, columns=columns)
    ser_test_writer = Data_Writer(folder, "ser_test.csv", index=Np_vect, columns=columns)
    mc_report = MC_Report_Writer(folder, "report.json", Nb_trials)
    mc_report.start()

    # Metric
    mse_criterion = MSELoss()
    ser_criterion = SERLoss()

    for Np in Np_vect:

        N_vect = np.arange(Np)  # preamble configuration
        demux = Demux(N_vect)
        mse_data_temp = np.zeros((2, nb_columns))
        ser_data_temp = np.zeros((2, nb_columns))
        mc_report.start_mc(Np)

        for indice in range(Nb_trials):

            # get dataset
            y, x_target = dataset[indice]
            demux.set_output(0)  # set,training data

            model_list = []
            # ------------------------------#
            #       clairvoyant detector    #
            # ------------------------------#
            model_clairvoyant = dataset.get_clairvoyant()

            # ------------------------------#
            #      classical approaches     #
            # ------------------------------#
            iq_reversed_parameters = [0.56061668, -0.07007708, -0.09110021, 1.26138753]

            # CFO and IQ known
            data_aided_fir1 = dsp.Data_Aided_FIR([1, 0, 0, 0, 0, 0, 0, 0])
            model_old1 = Sequential(OrderedDict([
                ('iq', channel.IQ_imbalance(iq_reversed_parameters)),
                ('cfo', channel.CFO(-0.005)),
                ('fir_est', data_aided_fir1),
                ('detector', dsp.Demodulator(alphabet))
                ]))

            x_out = model_old1(y)
            x_set = (demux(N_vect), demux(alphabet[x_target]))
            data_aided_fir1.fit(x_set)

            # IQ known
            data_aided_fir2 = dsp.Data_Aided_FIR([1, 0, 0, 0, 0, 0, 0, 0])
            model_old2 = Sequential(OrderedDict([
                ('iq', channel.IQ_imbalance(iq_reversed_parameters)),
                ('cfo', dsp.Blind_CFO(w0=0.005, method="newton")),
                ('fir_est', data_aided_fir2),
                ('detector', dsp.Demodulator(alphabet))
                ]))

            x_out = model_old2(y)
            x_set = (demux(N_vect), demux(alphabet[x_target]))
            data_aided_fir2.fit(x_set)

            # CFO known
            data_aided_fir3 = dsp.Data_Aided_FIR([1, 0, 0, 0, 0, 0, 0, 0])
            model_old3 = Sequential(OrderedDict([
                ('iq', dsp.Blind_IQ()),
                ('cfo', channel.CFO(-0.005)),
                ('fir_est', data_aided_fir3),
                ('detector', dsp.Demodulator(alphabet))
                ]))

            x_out = model_old3(y)
            x_set = (demux(N_vect), demux(alphabet[x_target]))
            data_aided_fir3.fit(x_set)

            # all unknown
            data_aided_fir4 = dsp.Data_Aided_FIR([1, 0, 0, 0, 0, 0, 0, 0])
            model_old4 = Sequential(OrderedDict([
                ('iq', dsp.Blind_IQ()),
                ('cfo', dsp.Blind_CFO(w0=0.005, method="newton")),
                ('fir_est', data_aided_fir4),
                ('detector', dsp.Demodulator(alphabet))
                ]))

            x_out = model_old4(y)
            x_set = (demux(N_vect), demux(alphabet[x_target]))
            data_aided_fir4.fit(x_set)

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

            x_set = (demux(N_vect), demux(x_target))
            trainer = Semi_Supervised_Trainer(model)
            model_phycom2, model_phycom1 = trainer.train(y, x_set)

            # perform testing
            model_list = [model_clairvoyant, model_old1, model_old2, model_old3, model_old4, model_phycom1, model_phycom2]

            for index_output in range(2):

                demux.set_output(index_output)
                x_target_temp = demux(x_target)

                for data_index, model_temp in enumerate(model_list):
                    x_pred = model_temp(y)
                    x_pred_temp = demux(x_pred)
                    x_pred_pre_detector = model_temp.get_data("detector")
                    x_pred_pre_detector_temp = demux(x_pred_pre_detector)

                    mse_data_temp[index_output][data_index+1] += mse_criterion(x_pred_pre_detector_temp, alphabet[x_target_temp])
                    ser_data_temp[index_output][data_index+1] += ser_criterion(x_pred_temp, x_target_temp)

            mc_report.update_current_mc(indice)

        mc_report.stop_current_mc()

        # compute mean
        mse_data_temp = mse_data_temp/M_testing
        ser_data_temp = ser_data_temp/M_testing
        mse_data_temp[0][0] = dataset.mse_theo(N_tot)
        mse_data_temp[1][0] = dataset.mse_theo(N_tot)
        ser_data_temp[0][0] = float("nan")
        ser_data_temp[1][0] = float("nan")

        # print data
        print("# Np={}".format(Np))
        print("MSE test: {}".format(mse_data_temp[1]))
        print("SER test: {}".format(ser_data_temp[1]))

        # save data
        mse_train_writer.add(mse_data_temp[0])
        mse_test_writer.add(mse_data_temp[1])
        ser_train_writer.add(ser_data_temp[0])
        ser_test_writer.add(ser_data_temp[1])

    # save data
    mc_report.stop()
    print("file saved in the folder results/simulation2")
