#!/usr/bin/env python3

#
# This example writes data to the existing empty dataset created by h5_crtdat.py and then reads it back.
#
import h5py
import numpy as np

#
# Open an existing file using default properties.
#
file = h5py.File('dset.h5', 'r+')
#
# Open "dset" dataset under the root group.
#
dataset = file['/dset']
#
# Initialize data object with 0.
#
data = np.zeros((4, 6))
#
# Assign new values
#
for i in range(4):
	for j in range(6):
		data[i][j] = i * 6 + j + 1
#
# Write data
#
print("Writing data...")
dataset[...] = data
#
# Read data back and print it.
#
print("Reading data back...")
data_read = dataset[...]
print("Printing data...")
print(data_read)
#
# Close the file before exiting
#
file.close()
