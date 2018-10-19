#!/usr/bin/env python3
import os

import numpy as np
import tensorflow as tf

from src.commons.constants import PROJECT_ROOT
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
	parser.add_argument("--sample_length", default=36, type=int,
	                    help="Length of 1 sample (TFRecord), i.e. number of nodes at the trunk level for IIGS6.")
	args = parser.parse_args()

	dataset_files = get_files_in_directory_recursively(
		rootdir=os.path.join(PROJECT_ROOT, 'src', 'nn', 'features', 'goofspiel', 'IIGS3',
		                     'tfrecord_dataset_IIGS3_1_3_false_true_lvl7')
	)

	trainset_ratio = 0.8
	devset_ratio = 0.1
	dataset_size = len(dataset_files)
	split_train = int(trainset_ratio * dataset_size)
	split_dev = int((trainset_ratio + devset_ratio) * dataset_size)

	training_set_dataset_files = dataset_files[0:split_train]
	dev_set_dataset_files = dataset_files[split_train:]

	train_dataset = DatasetFromTFRecord(
		batch_size=args.batch_size,  # 8
		dataset_files=training_set_dataset_files,
		sample_length=args.sample_length,  # 36
		variable_scope_name='train_dataset'
	)
	dev_dataset = DatasetFromTFRecord(
		batch_size=args.batch_size,  # 8
		dataset_files=dev_set_dataset_files,
		sample_length=args.sample_length,  # 36
		variable_scope_name='test_dataset'
	)

	with tf.Session() as sess:
		for epoch in range(args.epochs):
			print('Epoch', epoch)
			print('\tTrain set:')
			for sample in train_dataset.next_batch(sess):
				feature_input, feature_target = sample
				print(feature_input, feature_target)
			print('\tDev (Validation) set:')
			for sample in dev_dataset.next_batch(sess):
				feature_input, feature_target = sample
				print(feature_input, feature_target)
