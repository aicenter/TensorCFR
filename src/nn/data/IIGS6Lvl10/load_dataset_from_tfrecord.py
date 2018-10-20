#!/usr/bin/env python3
import os

import numpy as np
import tensorflow as tf

from src.nn.data.DatasetFromTFRecord import DatasetFromTFRecord
from src.utils.other_utils import get_files_in_directory_recursively

if __name__ == "__main__":
	import argparse

	np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)
	script_directory = os.path.dirname(os.path.abspath(__file__))
	np.random.seed(42)

	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
	parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
	parser.add_argument("--sample_length", default=14400, type=int,
	                    help="Length of 1 sample (TFRecord), i.e. number of nodes at the trunk level for IIGS6.")
	args = parser.parse_args()

	dataset_dir = "{}/tfrecords/tfrecord_dataset_IIGS6_1_6_false_true_lvl10".format(script_directory)
	dataset_files = get_files_in_directory_recursively(rootdir=dataset_dir)

	trainset_ratio = 0.8
	devset_ratio = 0.1
	dataset_size = len(dataset_files)
	split_train = int(trainset_ratio * dataset_size)
	split_dev = int((trainset_ratio + devset_ratio) * dataset_size)

	trainset_files = dataset_files[:split_train]
	devset_files = dataset_files[split_train:split_dev]
	testset_files = dataset_files[split_dev:]

	SHUFFLE_BATCHES = False
	trainset = DatasetFromTFRecord(
		batch_size=args.batch_size,  # 8
		dataset_files=trainset_files,
		sample_length=args.sample_length,  # 14400
		shuffle_batches=SHUFFLE_BATCHES,
		variable_scope_name='trainset'
	)
	devset = DatasetFromTFRecord(
		batch_size=args.batch_size,  # 8
		dataset_files=devset_files,
		sample_length=args.sample_length,  # 14400
		shuffle_batches=SHUFFLE_BATCHES,
		variable_scope_name='devset'
	)
	testset = DatasetFromTFRecord(
		batch_size=args.batch_size,  # 8
		dataset_files=testset_files,
		sample_length=args.sample_length,  # 14400
		shuffle_batches=SHUFFLE_BATCHES,
		variable_scope_name='testset'
	)

	with tf.Session() as sess:
		for epoch in range(args.epochs):
			print('Epoch #{}:'.format(epoch))

			print('\tTrainset:')
			for sample in trainset.next_batch(sess):
				features, targets = sample
				print('\t\tBatch #{}:'.format(trainset.batch_id))
				print("Features:\n{}".format(features))
				print("Targets:\n{}".format(targets))

			print('\tDevset:')
			for sample in devset.next_batch(sess):
				features, targets = sample
				print('\t\tBatch #{}:'.format(devset.batch_id))
				print("Features:\n{}".format(features))
				print("Targets:\n{}".format(targets))

		print('Testset:')
		cumulative_variance = None
		for sample in testset.next_batch(sess):
			features, targets = sample
			batch_variance = np.var(targets, axis=0)
			if cumulative_variance is None:
				cumulative_variance = batch_variance * batch_variance.shape[0]
			else:
				cumulative_variance += batch_variance * batch_variance.shape[0]
			print('\tBatch #{}:'.format(testset.batch_id))
			print("Batch-variance of targets:\n{}".format(batch_variance))
			print("Cumulative variance of targets:\n{}".format(cumulative_variance))
