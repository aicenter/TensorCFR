#!/usr/bin/env python3

import numpy as np


class DatasetFromNPZ:
	def __init__(self, filename, shuffle_batches=True):
		self._batch_id = 0
		with np.load(filename) as data:
			self._features = data["features"]
			self._targets = data["targets"]

		self._shuffle_batches = shuffle_batches
		self._permutation = np.random.permutation(len(self._features)) if self._shuffle_batches \
			else range(len(self._features))

	@property
	def batch_id(self):
		return self._batch_id

	@property
	def features(self):
		return self._features

	@property
	def targets(self):
		return self._targets

	def next_batch(self, batch_size):
		self._batch_id += 1
		batch_size = min(batch_size, len(self._permutation))
		batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
		return self._features[batch_perm], self._targets[batch_perm] if self._targets is not None else None

	def epoch_finished(self):
		if len(self._permutation) == 0:
			self._batch_id = 0
			self._permutation = np.random.permutation(len(self._features)) if self._shuffle_batches \
				else range(len(self._features))
			return True
		return False
