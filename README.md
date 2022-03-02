# PhyCOM: A Multi-Layer Parametric Network For Joint Linear Impairments Compensation and symbol detection 


This repo contains the python code for the following paper

* Vincent Choqueuse, Alexandru Frunza, Stéphane Azou, Pascal Morel. ["PhyCOM: A Multi-Layer Parametric Network for Joint Linear Impairments Compensation and Symbol Detection"](https://arxiv.org/abs/2203.00266), Arxiv 2203.00266

## Abstract

In this paper, we focus on the joint impairments compensation and symbol detection problem in communication systems. First, we introduce a new multi-layer channel model that represents the underlying physics of multiple impairments in communication systems. This model is composed of widely linear parametric layers that describe the input-output relationship of the front-end impairments and channel effects. Using this particular model, we show that the joint compensation and zero-forcing detection problem can be solved by a particular feedforward network called PhyCOM. Because of the small number of network parameters, a PhyCOM network can be trained efficiently using sophisticated optimization algorithms and a limited number of pilot symbols.

Numerical examples are provided to demonstrate the effectiveness of PhyCOM networks with communication systems corrupted by transmitter and receiver IQ imbalances, carrier frequency offset, finite impulse response channels, and transmitter and receiver phase noise distortions. Compared to conventional digital signal processing approaches, simulation results show that the proposed technique is much more flexible and offers better statistical performance both in terms of MSE and SER with a moderate increase of the computation complexity.


## Python Code 

### Requirements

* python > 3
* numpy
* scipy
* matplotlib
* pandas


### Generate Figures

The `\examples` folder contains several python codes for creating the article figures. All the results are stored in the `\results\simulation_{}` subfolder.

* `\simulation1.py`: Figure 6
* `\simulation2_mse_ser.py`: Figure 7
* `\simulation3_mse_ser.py`: Figure 8
* `\simulation4_mse_ser.py`: Figure 10

### Custom Layer

To create new custom linear layers, you have to code a new class that inherits from the trainable module base class. The inherited class should override three methods : `compute_grad`, `backward` and `compute_H`. See the file `phycom/module.py`.

For example, the code of the CFO trainable layer is given below

```

class CFO(Trainable_Module):

    def __init__(self, cfo=0, name="CFO"):
        self.name = name
        self.grad = None
        self.requires_grad = True
        self.training = True
        self.set_parameter(cfo)

    def compute_H(self, N):
        # compute the layer transfer matrix
        n_vect = np.arange(N)
        D = np.diag(np.exp(1j*self._parameters[0]*n_vect))
        H = np.block([[np.real(D), -np.imag(D)], [np.imag(D), np.real(D)]])
        return H

    def compute_grad(self):
        # compute gradient 
        N = len(self._x)
        n_vect = np.arange(N)
        term1 = 1j*n_vect*np.exp(1j*self._parameters[0]*n_vect)*self._x
        K = np.atleast_2d(term1).T
        L = np.vstack([np.real(K), np.imag(K)])
        return L

    def forward(self, x):
        # forward propagation
        N = len(x)
        n_vect = np.arange(N)
        y = x*np.exp(1j*self._parameters[0]*n_vect)

        if self._training:
            self._x = x
            self._y = y
            self._H = self.compute_H(N)

        return y
```