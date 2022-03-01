import sys
import numpy as np
from collections import OrderedDict

sys.path.insert(0, '../src')

import dsp
from loss import SERLoss, MSELoss
from module import Demux
from utils import MC_Report_Writer, Data_Writer
from phycom.model import PhyCOM
from phycom.trainer import Semi_Supervised_Trainer
from phycom.module import IQ_imbalance, Inverse_FIR, QS_Phase_Noise, Demodulator
from dataset import Chain2Dataset

if __name__ == '__main__':

    # simulation parameters
    N = 50
    N_tot = 500
    SNR = 30
    alphabet = dsp.get_alphabet("QAM", 16)
    M_testing = 10  # increase size to improve the estimation of the MSE / SER metrics
    folder = "../results/simulation4"
    Nk_vect = [0, 5, 10]

    # prepare MC simulations
    columns = ["theo", "clairvoyant"]
    for Nk in Nk_vect:
        columns.append("phycom1_{}".format(Nk))
        columns.append("phycom2_{}".format(Nk))
    nb_columns = len(columns)

    SNR_vect = range(0, 42, 2)
    N_vect = np.arange(N_tot)

    dataset = Chain2Dataset("data/test.csv", alphabet, N_tot, M_testing, snr_dB=0)
    Nb_trials = len(dataset)
    mse_train_writer = Data_Writer(folder, "mse_train.csv", index=SNR_vect, columns=columns)
    ser_train_writer = Data_Writer(folder, "ser_train.csv", index=SNR_vect, columns=columns)
    mse_test_writer = Data_Writer(folder, "mse_test.csv", index=SNR_vect, columns=columns)
    ser_test_writer = Data_Writer(folder, "ser_test.csv", index=SNR_vect, columns=columns)
    mc_report = MC_Report_Writer(folder, "report.json", Nb_trials)
    mc_report.start()

    # metrics
    mse_criterion = MSELoss()
    ser_criterion = SERLoss()

    pilot_index = np.arange(0, N_tot, int(N_tot/N)) # tracking 

    for SNR in SNR_vect:

        dataset.set_SNR(SNR)
        demux = Demux(pilot_index) # tracking configuration
        mse_data_temp = np.zeros((2, nb_columns))
        ser_data_temp = np.zeros((2, nb_columns))

        mc_report.start_mc(SNR)

        for indice in range(Nb_trials):

            # get dataset
            y, x_target = dataset[indice]
            demux.set_output(0)  # set,training data

            model_list = []
            # ------------------------------#
            #       clairvoyant detector    #
            # ------------------------------#
            model_clairvoyant = dataset.get_clairvoyant()
            model_list.append(model_clairvoyant)

            # ------------------------------#
            #       Phycom detector         #
            # ------------------------------#

            for N_k in Nk_vect:

                # phycom networks
                if N_k == 0:
                    model = PhyCOM(OrderedDict([
                            ('iq1', IQ_imbalance()),
                            ('fir', Inverse_FIR([1, 0, 0, 0, 0, 0, 0, 0])),
                            ('iq2', IQ_imbalance()),
                            ('detector', Demodulator(alphabet))
                            ]))
                else:
                    model = PhyCOM(OrderedDict([
                            ('iq1', IQ_imbalance()),
                            ('pn1', QS_Phase_Noise(np.zeros(N_k))),
                            ('fir', Inverse_FIR([1, 0, 0, 0, 0, 0, 0, 0])),
                            ('pn2', QS_Phase_Noise(np.zeros(N_k))),
                            ('iq2', IQ_imbalance()),
                            ('detector', Demodulator(alphabet))
                            ]))

                x_set = (demux(N_vect), demux(x_target))
                trainer = Semi_Supervised_Trainer(model)
                model_phycom2, model_phycom1 = trainer.train(y, x_set)
                model_list.append(model_phycom1)
                model_list.append(model_phycom2)

            # perform testing
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
        print("# SNR={} dB".format(SNR))
        print("MSE test: {}".format(mse_data_temp[1]))
        print("SER test: {}".format(ser_data_temp[1]))

        # save data
        mse_train_writer.add(mse_data_temp[0])
        mse_test_writer.add(mse_data_temp[1])
        ser_train_writer.add(ser_data_temp[0])
        ser_test_writer.add(ser_data_temp[1])

    # save data
    mc_report.stop()
    print("file saved in the folder results/simulation4")
