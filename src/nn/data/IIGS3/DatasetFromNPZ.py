#!/usr/bin/env python3
import numpy as np


class DatasetFromNPZ:
	def __init__(self, filename, shuffle_batches=True):
		with np.load(filename) as data:
			self._features = data["features"]
			self._targets = data["targets"]

		self._shuffle_batches = shuffle_batches
		self._permutation = np.random.permutation(len(self._features)) if self._shuffle_batches \
			else range(len(self._features))

	@property
	def features(self):
		return self._features

	@property
	def targets(self):
		return self._targets

	def next_batch(self, batch_size):
		batch_size = min(batch_size, len(self._permutation))
		batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
		return self._features[batch_perm], self._targets[batch_perm] if self._targets is not None else None

	def epoch_finished(self):
		if len(self._permutation) == 0:
			self._permutation = np.random.permutation(len(self._features)) if self._shuffle_batches \
				else range(len(self._features))
			return True
		return False


if __name__ == "__main__":
	import argparse

	# Fix random seed
	np.random.seed(42)

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
	parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
	parser.add_argument("--train_file", default="./IIGS3/train.npz",
	                    help="Filename of train dataset.")
	args = parser.parse_args()

	# Load the data
	# train = DatasetFromNPZ(args.train_file)
	train = DatasetFromNPZ("IIGS3_1_3_false_true_lvl7_numpy_dataset.npz")
	# dev = DatasetFromNPZ("fashion-masks-dev.npz")   TODO
	# test = DatasetFromNPZ("fashion-masks-test.npz", shuffle_batches=False) TODO

	batches_per_epoch = len(train._features) // args.batch_size

	for i in range(args.epochs):
		while not train.epoch_finished():
			features, targets = train.next_batch(args.batch_size)
			raise NotImplementedError("Show train dataset")  # TODO
		# raise NotImplementedError("Show dev dataset")  # TODO

	# with open("{}/fashion_masks_test.txt".format(args.logdir), "w") as test_file:
	# 	while not test.epoch_finished():
	# 		features, _ = test.next_batch(args.batch_size)
	# 		raise NotImplementedError("Show test dataset")  # TODO
