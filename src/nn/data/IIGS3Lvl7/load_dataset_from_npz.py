#!/usr/bin/env python3

import os

import numpy as np

from src.nn.data.DatasetFromNPZ import DatasetFromNPZ

if __name__ == "__main__":
	import argparse

	np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)
	script_directory = os.path.dirname(os.path.abspath(__file__))
	np.random.seed(42)

	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=3, type=int, help="Batch size.")
	parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
	args = parser.parse_args()

	train_file = "{}/80-10-10/IIGS3_1_3_false_true_lvl7_train.npz".format(script_directory)
	train = DatasetFromNPZ(train_file)

	for epoch in range(args.epochs):
		print("Epoch #{}:".format(epoch))
		while not train.epoch_finished():
			print("Batch #{}:".format(train.batch_id))
			features, targets = train.next_batch(args.batch_size)
			print("Features:\n{}".format(features))
			print("Targets:\n{}".format(targets))
