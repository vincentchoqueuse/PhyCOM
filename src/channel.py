import numpy as np
from module import Module, Linear_Module
from scipy.stats import norm
from scipy import signal
from scipy.linalg import toeplitz


def compute_awgn_ser(modulation, order, sigma2, sigma2s=1):

    M = order

    if modulation == "PAM":
        # See Proakis Book equation 5-2-45 on page 268
        ser = 2*((M-1)/M)*norm.sf(np.sqrt(6*sigma2s/(sigma2*(M*M-1))))

    if modulation == "PSK":

        termQ = norm.sf(np.sqrt(2*sigma2s/sigma2))
        if M == 2:
            # See Proakis Book equation 5-2-57 on page 271
            ser = termQ
        if M >= 4:
            # approximated value: See Proakis Book equation 5-2-61 on page 273
            ser = 2*norm.sf(np.sqrt(2*sigma2s/sigma2)*np.sin(np.pi/M))

    if modulation == "QAM":
        # See Proakis equation 5-2-78 / 5-2-79 on page 280"""
        PsqrtM = 2*(1-(1/np.sqrt(M)))*norm.sf(np.sqrt(3*sigma2s/(sigma2*(M-1))))
        ser = 1-((1-PsqrtM)**2)

    return ser


class SP(Module):

    r"""
    SP Module

    Parameters
    ----------
    name : string, optional
        Layer Name
    """

    def __init__(self, size, name="SP"):
        self.name = name
        self.size = size

    def forward(self, x):
        N = len(x)
        M = int(np.log2(self.size))

        # construct symbols from binary stream
        data = np.reshape(x, (M, int(N/M)), order="F")
        vect1 = 2**np.arange(M)
        y = np.matmul(vect1, data)
        return y


class PS(Module):
    r"""
    SP Module
    """

    def __init__(self, size, name="SP"):
        self.name = name
        self.size = size

    def forward(self, x):
        M = int(np.log2(self.size))
        binary_temp = 1*(((x[:, None] & (1 << np.arange(M)))) > 0)
        y = np.ravel(binary_temp, order="C")
        return y


class Noise(Module):

    def __init__(self, sigma2=0, is_complex=True, frozen=False, b=None, name="noise"):
        self.set_parameter(sigma2)
        self.name = name
        self.is_complex = is_complex
        self._frozen = frozen
        self._b = b

    def rvs(self, N):
        sigma2 = self._parameters[0]

        if self.is_complex:
            scale = np.sqrt(sigma2/2)
            b = norm.rvs(loc=0, scale=scale, size=N) + 1j*norm.rvs(loc=0, scale=scale, size=N)
        else:
            scale = np.sqrt(sigma2)
            b = norm.rvs(loc=0, scale=scale, size=N)
        return b

    def forward(self, x):
        N = len(x)

        if not self._frozen:
            self._b = self.rvs(N)

        y = x + self._b
        return y


class IQ_imbalance(Linear_Module):

    r"""
    IQ_imbalance()
    Physical Layer for IQ imbalance impairments. IQ imbalance is  modeled in the complex domain as

    .. math::
        x_l[n] = \alpha x_{l-1}[n]+ \beta x_{l-1}^*[n]

    where :math:`n=0,1\cdots,N-1` and :math:`(\alpha,\beta) \in \mathbb{C}^2.

    Parameters
    ----------
    params : numpy array
        A numpy array of size 4 containing the IQ imbalance parameters

    Returns
    -------
    None
    """

    def __init__(self, parameters, name="IQ Impairment layer"):
        self.name = name
        self.set_parameters(parameters)

    def forward(self, x):

        params = self._parameters
        alpha = (params[0]+params[3])/2+1j*(params[2]-params[1])/2
        beta = (params[0]-params[3])/2+1j*(params[1]+params[2])/2
        y = alpha*x + beta*np.conj(x)

        return y

    def get_reversed_parameters(self):
        # isomorphic layer
        params = np.copy(self._parameters)
        det = (params[0]*params[3]-params[1]*params[2])
        params_reverse = (1/det)*np.array([params[3], -params[1], -params[2], params[0]])
        return params_reverse


class CFO(Linear_Module):
    r"""
    CFO()

    The effect of a residual Carrier Frequency Offset (CFO) is usually modeled as \cite{YAO05}

    .. math::
        x_l[n] =x_{l-1}[n]e^{j\omega n}

    where :math:`\omega` corresponds to the normalized residual carrier offset (in rad/samples). The CFO layer only depends on the layer parameter $\boldsymbol\theta = \omega$.

    Parameters
    ----------
    params : numpy array
        Numpy array of length 1 containing the CFO parameters

    Returns
    -------
    y :  numpy array
         Output signal
    """
    def __init__(self, parameters, name="CFO"):
        self.name = name
        self.set_parameter(parameters)

    def forward(self, x):
        N = len(x)
        n_vect = np.arange(N)
        y = x*np.exp(1j*self._parameters[0]*n_vect)
        return y

    def get_reversed_parameter(self):
        w0 = self._parameters[0]
        return -w0


class FIR(Linear_Module):

    r"""
    Let us consider a Finite Impulse Response (FIR) channel with D taps.
    The output of a FIR channel layer is given by

    .. math::
        y_l[n] = \sum_{d=0}^{D-1} h_d y_{l-1}[n-d]

    """

    def __init__(self, h, name="fir"):
        h_tilde = np.hstack([np.real(h), np.imag(h)])
        self.name = name
        self.set_parameters(h_tilde)

    @property
    def L(self):
        return int(self.nb_parameters()/2)

    def compute_H(self, N):
        L = self.L
        theta = self._parameters[:L]+1j*self._parameters[L:]
        first_col = np.zeros(N, dtype=np.complex)
        first_col[:L] = theta
        M = toeplitz(first_col, np.zeros(N))
        H = np.block([[np.real(M), -np.imag(M)], [np.imag(M), np.real(M)]])
        return H

    def forward(self, x):
        L = self.L
        theta = self._parameters[:L] + 1j*self._parameters[L:]
        y = signal.lfilter(theta, 1, x)
        return y


class MIMO_Channel(Linear_Module):

    def __init__(self, H, name="mimo_channel"):
        self.name = name
        self._parameter = H

    def forward(self, x):
        N = len(x)
        H = self._parameter
        y_tilde = np.matmul(H, np.hstack([np.real(x), np.imag(x)]))
        y = y_tilde[:N] + 1j*y_tilde[N:]
        return y


class Phase_Noise(Linear_Module):

    def __init__(self, sigma2, frozen=False, b=None, name="phase_noise"):
        self.name = name
        self.set_parameter(sigma2)
        self._frozen = frozen
        self._b = b

    def rvs(self, N):
        sigma2 = self._parameters[0]
        scale = np.sqrt(sigma2)
        noise = norm.rvs(loc=0, scale=scale, size=N)
        return np.cumsum(noise)

    def forward(self, x):
        N = len(x)

        if not self._frozen:
            self._b = self.rvs(N)

        y = x * np.exp(1j*self._b)
        return y
