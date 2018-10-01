#!/usr/bin/env python3

import numpy as np

from src.utils.other_utils import one_hot

N_CLASSES = 3

if __name__ == '__main__':
	features = np.array(
		[
			[0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 1.21300e-03],
			[0.00000e+00, 0.00000e+00, 0.00000e+00, 2.00000e+00, 3.27100e-03],
			[0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 4.97500e-02]
		]
	)
	indices = (features[:, :4]).astype(int)
	one_hot_features = one_hot(indices, N_CLASSES)
	rows = indices.shape[0]
	columns = indices.shape[1]
	one_hot_flattened_features = one_hot_features.reshape(rows, columns * N_CLASSES)

	print("features:\n{}".format(features))
	print("indices:\n{}".format(indices))
	# print("one_hot_features:\n{}".format(one_hot_features))
	print("one_hot_flattened_eatures:\n{}".format(one_hot_flattened_features))
