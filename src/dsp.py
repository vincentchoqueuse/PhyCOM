from module import Module
import numpy as np
import numpy.linalg as LA
from scipy.linalg import toeplitz
from scipy.signal import deconvolve


def get_alphabet(type, order, normalize=True):

    if type == "PSK":
        alphabet = np.exp(2j*np.pi*np.arange(order)/order)

    if type == "QAM":
        if order == 16:
            alphabet = np.array([-3-3j, -3-1j, -3+3j, -3+1j, -1-3j, -1-1j, -1+3j, -1+1j, 3-3j, 3-1j, 3+3j, 3+1j, 1-3j, 1-1j, 1+3j, 1+1j])

    if normalize:
        coef = np.mean(np.abs(alphabet)**2)
        alphabet = (1/np.sqrt(coef))*alphabet

    return alphabet


class Modulator(Module):

    """
    Modulator()

    Modulate an array of integer using a particular real of complex-valued alphabet
    """

    def __init__(self, alphabet, name="modulator"):
        self._alphabet = alphabet
        self.name = name

    def set_alphabet(self, alphabet):
        self._alphabet = alphabet

    def forward(self, x):
        y = self._alphabet[x]
        return y


class Demodulator(Module):

    """
    Demodulator()

    Demodulate an array of integer using a particular real of complex-valued alphabet
    """

    def __init__(self, alphabet=[-1, 1], name="detector"):
        self._alphabet = alphabet
        self.name = name

    def set_alphabet(self, alphabet):
        self._alphabet = alphabet

    def get_alphabet(self):
        return self._alphabet

    def forward(self, x):
        x = np.transpose(np.atleast_2d(x))
        x = np.argmin(np.abs(x-self._alphabet)**2, axis=1)
        return x


class Blind_IQ(Module):

    """
    Blind_IQ()
    Blind IQ estimator based on the diagonalisation of the augmented covariance matrix. This compensation assumes the circularity of compensated signal


    Parameters
    ----------
    None

    Returns
    -------
    Compensated signal

    """

    def __init__(self, name="iq_compensator"):
        self.name = name

    def forward(self, x):

        N = len(x)
        X = np.vstack([np.real(x), np.imag(x)])

        # compute covariance matrix
        R = (1/N)*np.matmul(X, np.transpose(X))
        # perform eigenvalue decomposition
        V, U = LA.eig(R)

        # perform whitening
        D = np.diag(1/np.sqrt(V))
        M = np.matmul(D, np.transpose(U))
        Y = np.matmul(M, X)
        x = Y[0, :] + 1j*Y[1, :]
        return x


class Blind_CFO(Module):

    """
    Blind_CFO()
    Blind CFO estimator based on the maximisatio of the periodogram of 4th order statistic

    .. math::

        \\widehat{\\omega} = \\frac{1}{4} \\arg \\max_{\\omega} |\\sum_{n=0}^{N-1}x^4[n]e^{-j\\omega n}|^2

    The maximisation is performed using the Newton Algorithm

    Parameters
    ----------
    w0 : float
        Initialisation in rad/samples
    N_iter : int
        Number of iterations
    method : str

    Returns
    -------
    Compensated signal

    """

    def __init__(self, w0=0, N_iter=10, training=True, method="newton", step_size=10**(-5), name="cfo"):
        self.name = name
        self.w_init = w0
        self.N_iter = N_iter
        self.method = method
        self.step_size = step_size
        self._training = training

    def cost(self, x, w):
        N = len(x)
        x4 = x**4
        dtft = self.compute_dtft(x4, w)
        return (np.abs(dtft)**2)/N

    def compute_dtft(self, x, w):
        N = len(x)
        N_vect = np.arange(N)
        dtft = np.sum(x*np.exp(-1j*w*N_vect))
        return dtft

    def fit(self, x, w0):
        w = w0
        N = len(x)
        x4 = x**4
        N_vect = np.arange(N)
        step_size = self.step_size

        if self.method == "grid-search":
            w_vect = 4*np.arange(0.0045, 0.0055, 0.00001)
            cost_vect = np.zeros(len(w_vect))
            for index, w in enumerate(w_vect):
                cost_vect[index] = self.cost(x, w)
            index_max = np.argmax(cost_vect)
            w = w_vect[index_max]

        else:
            for n in range(self.N_iter):
                if self.method == "grad":
                    dtft = self.compute_dtft(x4, w)
                    dtft_diff = self.compute_dtft(-1j*N_vect*x4, w)
                    grad = (1/N) * (dtft_diff*np.conj(dtft) + dtft*np.conj(dtft_diff))
                    h = step_size * grad

                if self.method == "newton":
                    dtft = self.compute_dtft(x4, w)
                    dtft_diff = self.compute_dtft(-1j*N_vect*x4, w)
                    dtft_diff2 = self.compute_dtft(-(N_vect**2)*x4, w)
                    grad = (1/N) * (dtft_diff*np.conj(dtft) + dtft*np.conj(dtft_diff))
                    J = (2/N) * (np.real(dtft_diff2*np.conj(dtft)) + (np.abs(dtft_diff)**2))
                    h = -grad/J

                w = w + h

        w0 = np.real(w)/4
        return w0

    def forward(self, x):

        N = len(x)
        N_vect = np.arange(N)
        if self._training:
            w0 = self.fit(x, 4*self.w_init)
        x = x*np.exp(-1j*w0*N_vect)
        return x


class Data_Aided_FIR(Module):

    """
    Data_Aided_FIR()
    Data aided estimation of a FIR filter using ZF estimation.
    """

    def __init__(self, h, name="data_aided_fir"):
        self._h = h
        self.name = name

    def fit(self, y_set):
        # Estimation of the channel coefficient
        # x = Hy = Yh
        index_pilots = y_set[0]
        y_target = y_set[1]
        N = len(index_pilots)
        L = len(self._h)
        x_trunc = self._x[index_pilots]

        first_col = np.zeros(N, dtype=np.complex)
        first_col = y_target
        H = toeplitz(first_col, np.zeros(L))
        self._h = np.matmul(LA.pinv(H), x_trunc)

    def forward(self, x):
        L = len(self._h)
        y, _ = deconvolve(np.hstack([x, np.zeros(L-1)]), self._h)
        return y
