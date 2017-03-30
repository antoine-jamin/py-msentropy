# -*-coding:Latin-1 -*

__author__ = "Antoine JAMIN"

'''
To test the package we generate nb_signal white noise of N samples and we calculate the mse and the rcmse of each signal.
After we print the mean of the nb_signal mse and the nb_signal rcmse.
And we print also the complexity index for mse and rcmse and the cross-sample entropy value betweene white_noise[0] and
                                                                                                        white_noise[1]
'''

import msentropy as msen
import numpy as np
import matplotlib.pyplot as plt

# Variable definition
nb_signal = 10
N = 1000
nbscales = 12
m = 1

# nb_signal white noise of N samples generation
white_noise = []
for i in range(0, nb_signal):
    white_noise.append(np.random.normal(0, 1, size=N))

# MSE calculation for each white noise signal
MSE = np.zeros(nbscales + 1)
for j in range(0, nb_signal):
    signal = white_noise[j]
    MSE_temp = msen.mse(m, 0.15 * np.std(signal), signal, nbscales)
    for k in range(0, len(MSE_temp)):
        MSE[k] += MSE_temp[k]

# mean of the nb_signal MSE
MSE /= nb_signal

# RCMSE calculation for each white noise signal
RCMSE = np.zeros(nbscales + 1)
for k in range(0, nb_signal):
    signal = white_noise[j]
    RCMSE_temp = msen.rcmse(signal, m, 0.15 * np.std(signal), nbscales)
    for k in range(0, len(RCMSE_temp)):
        RCMSE[k] += RCMSE_temp[k]

# mean of the nb_signal RCMSE
RCMSE /= nb_signal

# Print the results
fig = plt.figure()
fig.add_subplot(211).plot(MSE, "b-o")
plt.title("MSE")

fig.add_subplot(212).plot(RCMSE, "r-o")
plt.title("RCMSE")

print("Complexity index of MSE = " + str(msen.complexity_index(MSE, 1, nbscales)))
print("Complexity index of RCMSE = " + str(msen.complexity_index(RCMSE, 1, nbscales)))

r = (0.15 * (np.std(white_noise[0] + np.std(white_noise[1])))) / 2
print(("Cross-sample entropy of white noise 0 and 1 : " + str(msen.cross_SampEn(white_noise[0], white_noise[1], m, r))))

plt.show()
