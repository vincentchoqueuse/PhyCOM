# PhyCOM: A Multi-Layer Parametric Network For Joint Linear Impairments Compensation and symbol detection 


This repo contains the python code for the following paper

* Vincent Choqueuse, Alexandru Frunza, Stéphane Azou, Pascal Morel. "PhyCOM: A Multi-Layer Parametric Network for Joint Linear Impairments Compensation and Symbol Detection". 

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


