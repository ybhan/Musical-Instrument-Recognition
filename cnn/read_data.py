# -*- coding: utf-8 -*-
""" By HAN Yuanbo, 2018-11-15."""
import scipy.io as sio
import numpy as np

instruments = ('XY',)  # All the instruments
num_instru = len(instruments)

x = [[]] * num_instru
y = [[]] * num_instru

for i in range(num_instru):
    # Read in .mat
    instru = instruments[i]
    mat = sio.loadmat('../mat_files/'+instru+'.mat')

    x[i] = np.array(mat[instru])
    y[i] = np.ones(shape=(x[i].shape[0]), dtype=np.long) * i  # Label = i


data_set = np.concatenate(x, axis=0)
target_set = np.concatenate(y, axis=0)

# Save data
np.save("data_set.npy", data_set)
np.save("target_set.npy", target_set)
