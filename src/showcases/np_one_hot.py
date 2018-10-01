#!/usr/bin/env python3

import numpy as np

N_CLASSES = 3


def one_hot(a, num_classes):
	return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


if __name__ == '__main__':
	features = np.array(
		[
			[0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 1.21300e-03],
			[0.00000e+00, 0.00000e+00, 0.00000e+00, 2.00000e+00, 3.27100e-03],
			[0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 4.97500e-02]
		]
	)
	indices = (features[:, :4]).astype(int)
	print(features)
	print(indices)
	print(one_hot(indices, N_CLASSES))
