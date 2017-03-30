# -*-coding:Latin-1 -*
__author__ = "Antoine JAMIN"


'''
This python package was developed during my internship at LARIS (http://laris.univ-angers.fr/fr/index.html)
This python package implements some function to calculate multi-scale entropy, refined composite multi-scale entropy
and cross-sample entropy.
To develop this package we use these references :
    [S M. Pincus, 1991] -- Approximate entropy as mesure of system complexity.
    [J. Richman, 2000] -- Physiological time-series analysis using approximate entropy and sample entropy.
    [A. Humeau, 2015] -- The Multiscale Entropy Algorithm and Its Variants : A Review.
    [M. Costa,2002] -- Multiscale Entropy Analysis of Complex Physiologic Time Series.
    [S. Wu, 2014] -- Analysis of complex time series using refined composite multiscale entropy
    [Y. Chang, 2014] -- Application of a Modified Entropy Computational Method in Assessing the Complexity of Pulse
                        Wave Velocity Signals in Healthy and Diabetic Subjects.
    [D. Kong, 2011] -- Use of modified sample entropy measurement to classify ventricular tachycardia and fibrillation.
    [T. Zhang, 2007] -- Cross-sample entropy statistic as a measure of complexity and regularity of renal sympathetic
                        nerve activity in the rat.
    [W. Shi, 2013] -- Cross-sample entropy statistic as a measure of synchronism and cross-correlation of stock markets.
    [C. C. Chiu, 2011] -- Assessment of Diabetics with Various Degrees of Autonomic Neuropathy Based on
                            Cross-Approximate Entropy
This package use the pyeeg package you need to import it (Copyleft 2010 Forrest Sheng Bao http://fsbao.net) :
                                                                        http://pyeeg.sourceforge.net
This package use the numpy package you need to import it (http://numpy.scipy.org)
'''

import numpy as np
import pyeeg


# Variables globales
nb_scales = 20
length_sample = 1000




## Coarse graining procedure
# tau : scale factor
# signal : original signal
# return the coarse_graining signal
def coarse_graining(tau, signal):
    # signal lenght
    N = len(signal)
    # Coarse_graining signal initialisation
    y = np.zeros(int(len(signal) / tau))
    for j in range(0, int(N / tau)):
        y[j] = sum(signal[i] / tau for i in range(int((j - 1) * tau), int(j * tau)))
    return y


## Multi-scale entropy
# m : length of the patterns that compared to each other
# r : tolerance
# signal : original signal
# return the Multi-scale entropy of the original signal (array of nbscales length)
def mse(m, r, signal, nbscales=None):
    # Output initialisation
    if nbscales == None:
        nbscales = int((len(signal) * nb_scales) / length_sample)
    y = np.zeros(nbscales + 1)
    y[0] = float('nan')
    for i in range(1, nbscales + 1):
        y[i] = pyeeg.samp_entropy(coarse_graining(i, signal), m, r)
    return y


## calculation of the matching number
# it use in the refined composite multi-scale entropy calculation
def match(signal, m, r):
    N = len(signal)

    Em = pyeeg.embed_seq(signal, 1, m)
    Emp = pyeeg.embed_seq(signal, 1, m + 1)

    Cm, Cmp = np.zeros(N - m - 1) + 1e-100, np.zeros(N - m - 1) + 1e-100
    # in case there is 0 after counting. Log(0) is undefined.

    for i in range(0, N - m):
        for j in range(i + 1, N - m):  # no self-match
            # if max(abs(Em[i]-Em[j])) <= R:  # v 0.01_b_r1
            if pyeeg.in_range(Em[i], Em[j], r):
                Cm[i] += 1
                # if max(abs(Emp[i] - Emp[j])) <= R: # v 0.01_b_r1
                if abs(Emp[i][-1] - Emp[j][-1]) <= r:  # check last one
                    Cmp[i] += 1

    return sum(Cm), sum(Cmp)


## Refined Composite Multscale Entropy
# signal : original signal
# m : length of the patterns that compared to each other
# r : tolerance
# nbscales :
# return the RCMSE of the original signal (array of nbscales length)
def rcmse(signal, m, r, nbscales):
    Nm = 0
    Nmp = 0
    y = np.zeros(nbscales + 1)
    y[0] = float('nan')
    for i in range(1, nbscales + 1):
        for j in range(0, i):
            (Cm, Cmp) = match(coarse_graining(i, signal[i:]), m, r)
            Nm += Cm
            Nmp += Cmp
        y[i] = -np.log(Nmp / Nm)
    return y


## Caclulate the complexity index of the MSE (or RCMSE) of the original signal
# sig : RCMSE or MSE of the original signal
# inf : lower bound for the calcul
# sup : upper bound for the calcul
# return the complexity index value
def complexity_index(sig, low, upp):
    ci = sum(sig[low:upp])
    return ci


## Calculate the cross-sample entropy of 2 signals
# u : signal 1
# v : signal 2
# m : length of the patterns that compared to each other
# r : tolerance
# return the cross-sample entropy value
def cross_SampEn(u, v, m, r):
    B = 0.0
    A = 0.0
    if (len(u) != len(v)):
        raise Exception("Error : lenght of u different than lenght of v")
    N = len(u)
    for i in range(0, N - m):
        for j in range(0, N - m):
            B += cross_match(u[i:i + m], v[j:j + m], m, r) / (N - m)
            A += cross_match(u[i:i + m + 1], v[j:j + m + 1], m + 1, r) / (N - m)
    B /= N - m
    A /= N - m
    cse = -np.log(A / B)
    return cse


## calculation of the matching number
# it use in the cross-sample entropy calculation
def cross_match(signal1, signal2, m, r):
    # return 0 if not match and 1 if match
    d = []
    for k in range(0, m):
        d.append(np.abs(signal1[k] - signal2[k]))
    if max(d) <= r:
        return 1
    else:
        return 0