#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py
import logging

import numpy as np
import tensorflow as tf

from src.nn.Runner_CNN_IIGS6Lvl10_NPZ import Runner_CNN_IIGS6Lvl10_NPZ
from src.nn.data.DatasetFromTFRecord import DatasetFromTFRecord
from src.utils.other_utils import get_files_in_directory_recursively

# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = False


class Runner_CNN_IIGS6Lvl10_TFRecords(Runner_CNN_IIGS6Lvl10_NPZ):
	def __init__(self, fixed_randomness=False):
		super().__init__(fixed_randomness)
		self.data_session = None
		self.SAMPLE_LENGTH = 14400

	def add_arguments_to_argparser(self):
		super().add_arguments_to_argparser()
		self.argparser.add_argument("--trainset_ratio", default=0.8, type=float, help="Ratio of dataset for trainset.")
		self.argparser.add_argument("--devset_ratio", default=0.1, type=float, help="Ratio of dataset for devset.")

	def datasets_from_tfrecords(self, dataset_directory, dev_batch_size=None, test_batch_size=None):
		dataset_files = get_files_in_directory_recursively(rootdir=dataset_directory)

		dataset_size = len(dataset_files)
		split_train = int(self.args.trainset_ratio * dataset_size)
		split_dev = int((self.args.trainset_ratio + self.args.devset_ratio) * dataset_size)

		trainset_files = dataset_files[:split_train]
		devset_files = dataset_files[split_train:split_dev]
		testset_files = dataset_files[split_dev:]

		trainset = DatasetFromTFRecord(
			batch_size=self.args.batch_size,
			dataset_files=trainset_files,
			sample_length=self.SAMPLE_LENGTH,
			shuffle_batches=True,
			variable_scope_name='trainset'
		)
		devset = DatasetFromTFRecord(
			batch_size=self.args.batch_size,
			dataset_files=devset_files,
			sample_length=self.SAMPLE_LENGTH,
			shuffle_batches=False,
			variable_scope_name='devset'
		)
		testset = DatasetFromTFRecord(
			batch_size=self.args.batch_size,
			dataset_files=testset_files,
			sample_length=self.SAMPLE_LENGTH,
			shuffle_batches=False,
			variable_scope_name='testset'
		)
		return devset, testset, trainset

	def init_datasets(self, dataset_directory):
		devset, testset, trainset = self.datasets_from_tfrecords(
			dataset_directory=dataset_directory
		)
		return devset, testset, trainset

	def train_one_epoch(self, trainset):
		for sample in trainset.next_batch(self.data_session):
			print("[epoch #{}, batch #{}] Training...".format(self.epoch, trainset.batch_id))
			reaches, targets = sample
			self.network.train(reaches, targets)

	def evaluate_devset(self, devset):
		for sample in devset.next_batch(self.data_session):
			reaches, targets = sample
			devset_error_mse, devset_error_infinity = self.network.evaluate("dev", reaches, targets)
			print("[epoch #{}, dev batch #{}] dev MSE {}, \tdev L-infinity error {}".format(
				self.epoch, devset.batch_id, devset_error_mse, devset_error_infinity))

	def evaluate_testset(self, testset):
		for sample in testset.next_batch(self.data_session):
			reaches, targets = sample
			testset_error_mse, testset_error_infinity = self.network.evaluate("test", reaches, targets)
			print("[epoch #{}, test batch #{}] test MSE {}, \ttest L-infinity error {}".format(
				self.epoch, testset.batch_id, testset_error_mse, testset_error_infinity))

	def showcase_predictions(self, trainset):
		print("Showcase of predictions:")
		reps = 100
		reaches = np.random.rand(1, self.SAMPLE_LENGTH)
		logging.debug("\treaches: {}".format(reaches))
		predictions = np.zeros((reps, self.SAMPLE_LENGTH))
		for i in range(reps):
			predictions[i] = self.network.predict(reaches)
			logging.info("prediction[{}]:\n\t {}\n".format(i, predictions[i]))
		variances = np.var(predictions, axis=0)

		# statistics
		print("min of variances: {}".format(np.amin(variances)))
		print("mean of variances: {}".format(np.mean(variances)))
		print("median of variances: {}".format(np.median(variances)))
		print("max of variances: {}".format(np.amax(variances)))
		# uncomment to plot:
		# plt.hist(variances, bins='auto')
		# plt.title("Histogram of variances")
		# plt.show()


	def run_neural_net(self, ckpt_every=None, ckpt_dir=None):
		with tf.Session() as self.data_session:
			super().run_neural_net(ckpt_every, ckpt_dir)

	def run_neural_net_from_ckpt(self, ckpt_dir=None, ckpt_basename=None):
		if (ckpt_dir is None) or (ckpt_basename is None):
			ckpt_dir = self.args.ckpt_dir
			ckpt_basename = self.args.ckpt_basename
		with tf.Session() as self.data_session:
			super().run_neural_net_from_ckpt(ckpt_dir, ckpt_basename)


if __name__ == '__main__' and ACTIVATE_FILE:
	runner = Runner_CNN_IIGS6Lvl10_TFRecords()
	runner.run_neural_net(ckpt_every=2)

	runner_from_ckpt = Runner_CNN_IIGS6Lvl10_TFRecords()
	# runner_from_ckpt.run_neural_net_from_ckpt(ckpt_dir=runner.args.logdir, ckpt_basename=runner.ckpt_basenames[-1])

	# Note: you can test this on:
	# i.e.
	# --ckpt_dir "logs/Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_204734-bs=32,ce=2,dr=0.1,e=3,e=C-46,r=C-46,t=1,tr=0.8"
	# --ckpt_basename "final_2018-11-11_20:47:52.ckpt"
	runner_from_ckpt.run_neural_net_from_ckpt()
