import numpy as np
import numpy.linalg as LA
from scipy.linalg import toeplitz
from module import Linear_Module


class Trainable_Module(Linear_Module):

    def __init__(self, parameters, name="layer"):
        self.name = name
        self.requires_grad = True
        self._training = True
        self.grad = None
        self.set_parameters(parameters)

    def backward(self, H_input):

        # accumulate gradient
        if self.requires_grad:
            L = self.compute_grad()
            grad = np.matmul(H_input, L)

            # accumulate grad
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

        # backpropagate matrix
        H_output = np.matmul(H_input, self._H)
        return H_output

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def zero_grad(self):
        self.grad = None


class IQ_imbalance(Trainable_Module):

    r"""
    IQ_imbalance()
    Physical Layer for IQ imbalance impairments.
    IQ imbalance is  modeled in the complex domain as

    .. math::
        x_l[n] = \alpha x_{l-1}[n]+ \beta x_{l-1}^*[n]

    where :math:`n=0,1\cdots,N-1` and :math:`(\alpha,\beta) \in \mathbb{C}^2`. This layer is parametrized by 4 real-valued nuisance parameters, which are given by

    .. math::

        \boldsymbol \theta =
        \begin{bmatrix}
        \theta_1\\
        \theta_2\\
        \theta_3\\
        \theta_4
        \end{bmatrix}=
        \begin{bmatrix}
        \Re e(\alpha+\beta)\\
        \Im m(-\alpha+\beta)\\
        \Im m(\alpha+\beta)\\
        \Re e(\alpha-\beta)
        \end{bmatrix}.

    Parameters
    ----------
    params : numpy array
        A numpy array of size 4 containing the IQ imbalance parameters
    Returns
    -------
    None
    """

    def __init__(self, parameters=[1, 0, 0, 1], name="IQ Compensation layer"):
        self.name = name
        self.grad = None
        self.requires_grad = True
        self.training = True
        self.set_parameters(parameters)

    def compute_H(self, N):

        r"""
        This method computes the layer transfer matrix. This matrix is given by

        .. math::
            \mathbf{H}(\boldsymbol \theta)=\begin{bmatrix}
            \theta_1& \theta_2\\
            \theta_3&\theta_4
            \end{bmatrix} \otimes \mathbf{I}_N.

        """
        G = np.reshape(self._parameters, (2, 2))
        H = np.kron(G, np.eye(N))
        return H

    def compute_grad(self):
        r"""
        This method compute the local Jacobian of the layer.
        The local jacobian is given by

        .. math::

            \mathbf{L}(\boldsymbol \theta) = \mathbf{I}_2 \otimes \begin{bmatrix}\Re e(\mathbf{x})&\Im m(\mathbf{x})\end{bmatrix}

        where :math:`\mathbf{x}` corresponds to the layer input

        """
        X = np.zeros((len(self._x), 2))
        X[:, 0] = np.real(self._x)
        X[:, 1] = np.imag(self._x)
        L = np.kron(np.eye(2), X)
        return L

    def forward(self, x):

        N = len(x)
        params = self._parameters
        alpha = (params[0]+params[3])/2 + 1j*(params[2]-params[1])/2
        beta = (params[0]-params[3])/2 + 1j*(params[1]+params[2])/2
        y = alpha*x + beta*np.conj(x)

        if self._training:
            self._x = x
            self._y = y
            self._H = self.compute_H(N)

        return y


class CFO(Trainable_Module):

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
    def __init__(self, cfo=0, name="CFO"):
        self.name = name
        self.grad = None
        self.requires_grad = True
        self.training = True
        self.set_parameter(cfo)

    def compute_H(self, N):

        r"""
        This method computes the layer transfer matrix. This matrix is given by

        .. math::
            \mathbf{H}(\boldsymbol \theta)=\begin{bmatrix}
            \Re e(\mathbf{M}(\boldsymbol \theta)) & -\Im m(\mathbf{M}(\boldsymbol \theta)) \\
            \Im m(\mathbf{M}(\boldsymbol \theta)) &\Re e(\mathbf{M}(\boldsymbol \theta))
            \end{bmatrix}

        where

        .. math::
            \mathbf{M}(\boldsymbol \theta)=
            \begin{bmatrix}
            1 & 0 &\cdots & 0 \\
            0 & e^{j \omega} &  &\vdots\\
            \vdots& & \ddots& 0\\
            0&\cdots&0&e^{j \omega(N-1)}
            \end{bmatrix}

        """
        n_vect = np.arange(N)
        D = np.diag(np.exp(1j*self._parameters[0]*n_vect))
        H = np.block([[np.real(D), -np.imag(D)], [np.imag(D), np.real(D)]])
        return H

    def compute_grad(self):
        r"""
        This method compute the local Jacobian of the layer.
        The local jacobian is given by

        .. math::
            \mathbf{L}(\boldsymbol \theta)= \begin{bmatrix}
            \Re e( \mathbf{K}(\boldsymbol \theta) )  \\
            \Im m(\mathbf{K}(\boldsymbol \theta) )
            \end{bmatrix}\label{eq_local_jacob2}

        where

        .. math::
            \mathbf{K}(\boldsymbol \theta)= \begin{bmatrix}
                0 \\
                je^{j \omega} \\
                \vdots \\
                j(N-1)e^{j \omega(N-1)}
                \end{bmatrix}\odot \mathbf{x}

        and :math:`\mathbf{x}` corresponds to the layer input.

        """
        N = len(self._x)
        n_vect = np.arange(N)
        term1 = 1j*n_vect*np.exp(1j*self._parameters[0]*n_vect)*self._x
        K = np.atleast_2d(term1).T
        L = np.vstack([np.real(K), np.imag(K)])
        return L

    def forward(self, x):
        N = len(x)
        n_vect = np.arange(N)
        y = x*np.exp(1j*self._parameters[0]*n_vect)

        if self._training:
            self._x = x
            self._y = y
            self._H = self.compute_H(N)

        return y


class Inverse_FIR(Trainable_Module):

    def __init__(self, h, name="inverse fir"):
        self.name = name
        self.grad = None
        self.requires_grad = True
        self.training = True
        parameters = np.hstack([np.real(h), np.imag(h)])
        self.set_parameters(parameters)

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

        try:
            H = LA.inv(H)
        except:
            pass

        return H

    def compute_grad(self):
        N = len(self._x)
        L = self.L
        Y = toeplitz(self._y, np.zeros(L))
        M2 = np.kron(np.array([1, 1j]), Y)
        M1 = self._H[:N, :N] + 1j*self._H[N:, :N]
        K = -1*np.matmul(M1, M2)
        L = np.block([[np.real(K)], [np.imag(K)]])
        return L

    def forward(self, x):

        N = len(x)
        H = self.compute_H(N)
        x_real = np.hstack([np.real(x), np.imag(x)])
        y_real = np.matmul(H, x_real)
        y = y_real[:N] + 1j*y_real[N:]

        if self._training:
            self._x = x
            self._y = y
            self._H = H

        return y


class QS_Phase_Noise(Trainable_Module):

    r"""
    QS_Phase_noise()

    The effect of a quasi static phase noise of length

    .. math::

        x_l[n] =x_{l-1}[n]e^{j\varphi[n]}

    where the phase :math:`\varphi[n]` is constant over K samples.

    Parameters
    ----------
    params : numpy array
        Numpy array of length K containing the initial phase
    Returns
    -------
    y :  numpy array
         Output signal


    """

    def __init__(self, parameters, name="CFO"):
        self._K = len(parameters)
        self.name = name
        self.grad = None
        self.requires_grad = True
        self.training = True
        self.set_parameters(parameters)

    def compute_H(self, N):

        r"""
        This method computes the layer transfer matrix. This matrix is given by

        .. math::

            \mathbf{H}(\boldsymbol \theta)=\begin{bmatrix}
            \Re e(\mathbf{M}(\boldsymbol \theta)) & -\Im m(\mathbf{M}(\boldsymbol \theta)) \\
            \Im m(\mathbf{M}(\boldsymbol \theta)) &\Re e(\mathbf{M}(\boldsymbol \theta))
            \end{bmatrix}

        where

        .. math::
            \mathbf{M}(\boldsymbol \theta) =
            \begin{bmatrix}
            e^{j \varphi_1} & 0 &\cdots & 0 \\
            0 & e^{j \varphi_2} &  &\vdots\\
            \vdots& & \ddots& 0\\
            0&\cdots&0&e^{j \varphi_K}
            \end{bmatrix}\otimes \mathbf{I}_{N/K}

        """
        I_mat = np.eye(int(N/self._K))
        D = np.diag(np.exp(1j*self._parameters))
        M = np.kron(D, I_mat)
        H = np.block([[np.real(M), -np.imag(M)], [np.imag(M), np.real(M)]])
        return H

    def compute_grad(self):
        r"""
        This method compute the local Jacobian of the layer.
        The local jacobian is given by

        .. math::

            \mathbf{L}(\boldsymbol \theta) =
            \begin{bmatrix}\Re e( \mathbf{K}(\boldsymbol \theta) )
            \\\Im m(\mathbf{K}(\boldsymbol \theta) )\end{bmatrix}

        where

        .. math::
            \mathbf{K}(\boldsymbol \theta)= \begin{bmatrix}
                0 \\
                je^{j \omega} \\
                \vdots \\
                j(N-1)e^{j \omega(N-1)}
                \end{bmatrix}\odot \mathbf{x}

        and :math:`\mathbf{x}` corresponds to the layer input.

        """
        N = len(self._x)
        d = np.exp(1j*self._parameters)
        K = np.zeros((N, self._K), dtype=np.complex)
        u = np.ones((int(N / self._K), 1))

        for indice in range(self._K):
            e = np.zeros(self._K)
            e[indice] = 1
            term1  = np.atleast_2d(1j * e * d).T
            k_n = np.ravel(np.kron(term1, u))
            K[:, indice] = k_n * (self._x)

        L = np.vstack([np.real(K), np.imag(K)])
        return L

    def forward(self, x):
        N = len(x)
        I_mat = np.ones(int(N/self._K))
        varphi = np.kron(self._parameters, I_mat)
        y = x*np.exp(1j*varphi)

        if self._training:
            self._x = x
            self._y = y
            self._H = self.compute_H(N)

        return y


class Demodulator(Trainable_Module):

    r"""
    Demodulator()
    Demodulator for the PhyCOM network that inherit from
    the Trainable Module Class

    Parameters
    ----------
    alphabet : numpy array
        Numpy array containing the symbol of the constellation
    Returns
    -------
    None
    """
    def __init__(self, alphabet=[-1, 1], name="detector"):
        self._alphabet = alphabet
        self.name = name
        self._parameters = []

    def set_alphabet(self, alphabet):
        self._alphabet = alphabet

    def get_alphabet(self):
        return self._alphabet

    def backward(self, H_input):
        return H_input

    def forward(self, x):
        x = np.transpose(np.atleast_2d(x))
        x = np.argmin(np.abs(x-self._alphabet)**2, axis=1)
        return x
