import numpy as np
import sys
import numpy.linalg as LA

sys.path.insert(0, '../src')

from collections import OrderedDict
from model import Sequential
from channel import Phase_Noise, Noise, IQ_imbalance, FIR, CFO, MIMO_Channel
from dsp import Modulator, Demodulator


class Dataset():

    def __init__(self, filename, alphabet, N, size, num_batches=None, snr_dB=30, name="dataset"):

        self.N = N
        self.size = size
        self.alphabet = alphabet
        self.name = name

        data = np.loadtxt("{}".format(filename), delimiter=",")
        if N > len(data):
            raise ValueError('Length of data is greater than list of frozen seeds!')
        else:
            self.seed_list = data[:size].astype(int)

        self.noise = Noise()
        self.modulator = Modulator(self.alphabet)
        self._model = self.get_model()
        self.set_SNR(snr_dB)

    def get_H(self, N):

        model = self._model
        model.set_frozen(True)
        self.modulator.set_bypass(True)
        self.noise.set_bypass(True)

        # sensing procedure
        H = np.zeros((2*N, 2*N))
        for indice in range(2*N):
            x = np.zeros(N, dtype=np.complex)
            if indice < N:
                x[indice] = 1
            else:
                x[indice-N] = 1j
            y = model(x)
            H[:, indice] = np.hstack([np.real(y), np.imag(y)])

        self.modulator.set_bypass(False)
        self.noise.set_bypass(False)
        model.set_frozen(False)
        return H

    def get_model(self):
        return None

    def get_clairvoyant(self):
        H = self.get_H(self.N)
        H_inv = LA.inv(H)

        clairvoyant_model = Sequential(OrderedDict([
            ('mimo', MIMO_Channel(H_inv)),
            ('detector', Demodulator(self.alphabet))
            ]))
        return clairvoyant_model

    def mse_theo(self, N):
        """ Compute performance of zero forcing detection """
        sigma2n = self.noise.get_parameter()
        H = self.get_H(self.N)

        H_inv = LA.inv(H)
        R = (sigma2n/2)*np.matmul(H_inv, np.transpose(H_inv))
        return 2*np.mean(np.diag(R))

    def set_SNR(self, snr_dB):
        sigma2s = np.mean(np.abs(self.alphabet)**2)
        sigma2n = sigma2s/(10**(snr_dB/10))
        self.noise.set_parameter(sigma2n)

    def __len__(self):
        return len(self.seed_list)

    def __getitem__(self, index):

        np.random.seed(seed=self.seed_list[index])
        order = len(self.alphabet)
        x = np.random.randint(low=0, high=order, size=self.N)
        y = self._model(x)
        return y, x


class Chain1Dataset(Dataset):

    h = np.array([0.9 + 0.1j, 0.3 + 0.3j, 0.1 + 0.05j, 0.02 + 0.1j, 0.1 - 0.05j, 0.02 - 0.1j, 0.1 + 0.03j, 0.04 - 0.012j])
    cfo = 0.005
    iq = [1.8, 0.1, 0.13, 0.8]

    def get_model(self):
        model = Sequential(OrderedDict([
            ('mod', self.modulator),
            ('fir', FIR(self.h)),
            ('cfo', CFO(self.cfo)),
            ('iq2', IQ_imbalance(self.iq)),
            ('noise', self.noise)
            ]))
        return model


class Chain2Dataset(Dataset):

    h = np.array([0.9 + 0.1j, 0.3 + 0.3j, 0.1 + 0.05j, 0.02 + 0.1j, 0.1 - 0.05j, 0.02 - 0.1j, 0.1 + 0.03j, 0.04 - 0.012j])
    iq1_params = [0.9, 0.4, -0.4, 0.6]
    iq2_params = [1.8, 0.1, 0.13, 0.8]
    sigma2_pn = 2*np.pi*(5*10**-5)

    def get_model(self):
        model = Sequential(OrderedDict([
            ('mod', self.modulator),
            ('iq1', IQ_imbalance(self.iq1_params)),
            ('pn1', Phase_Noise(self.sigma2_pn)),
            ('fir', FIR(self.h)),
            ('pn2', Phase_Noise(self.sigma2_pn)),
            ('iq2', IQ_imbalance(self.iq2_params)),
            ('noise', self.noise)
            ]))
        return model
