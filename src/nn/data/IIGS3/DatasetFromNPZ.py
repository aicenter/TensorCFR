#!/usr/bin/env python3

import os

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


if __name__ == "__main__":
	import argparse

	np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)
	script_directory = os.path.dirname(os.path.abspath(__file__))
	np.random.seed(42)

	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=3, type=int, help="Batch size.")
	parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
	args = parser.parse_args()

	train_file = "{}/train.npz".format(script_directory)
	train = DatasetFromNPZ(train_file)

	for epoch in range(args.epochs):
		print("Epoch #{}:".format(epoch))
		while not train.epoch_finished():
			print("Batch #{}:".format(train.batch_id))
			features, targets = train.next_batch(args.batch_size)
			print("Features:\n{}".format(features))
			print("Targets:\n{}".format(targets))
