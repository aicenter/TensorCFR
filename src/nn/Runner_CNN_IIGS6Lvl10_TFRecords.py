#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py
import tensorflow as tf

from src.nn.Runner_CNN_IIGS6Lvl10_NPZ import Runner_CNN_IIGS6Lvl10_NPZ
from src.nn.data.DatasetFromTFRecord import DatasetFromTFRecord
from src.nn.features.goofspiel.IIGS6.tfrecords_reaches_values_IIGS6_1_6_false_true_lvl10 import FEATURES_PER_FILE
from src.utils.other_utils import get_files_in_directory_recursively


class Runner_CNN_IIGS6Lvl10_TFRecords(Runner_CNN_IIGS6Lvl10_NPZ):   # TODO test parent here
	def __init__(self, fixed_randomness=False):
		super().__init__(fixed_randomness)
		self.data_session = None

	def datasets_from_tfrecords(self, script_directory, dataset_directory):
		dataset_dir = "{}/{}".format(script_directory, dataset_directory)
		dataset_files = get_files_in_directory_recursively(rootdir=dataset_dir)

		trainset_ratio = 0.8
		devset_ratio = 0.1
		dataset_size = len(dataset_files)
		split_train = int(trainset_ratio * dataset_size)
		split_dev = int((trainset_ratio + devset_ratio) * dataset_size)

		trainset_files = dataset_files[:split_train]
		devset_files = dataset_files[split_train:split_dev]
		testset_files = dataset_files[split_dev:]

		SAMPLE_LENGTH = 14400
		trainset = DatasetFromTFRecord(
			batch_size=self.args.batch_size,
			dataset_files=trainset_files,
			sample_length=SAMPLE_LENGTH,
			shuffle_batches=True,
			variable_scope_name='trainset'
		)
		devset = DatasetFromTFRecord(
			batch_size=len(devset_files) * FEATURES_PER_FILE,
			dataset_files=devset_files,
			sample_length=SAMPLE_LENGTH,
			shuffle_batches=False,
			variable_scope_name='devset'
		)
		testset = DatasetFromTFRecord(
			batch_size=self.args.batch_size,
			dataset_files=testset_files,
			sample_length=SAMPLE_LENGTH,
			shuffle_batches=False,
			variable_scope_name='testset'
		)
		return devset, testset, trainset

	def init_datasets(self, dataset_directory):
		import os
		script_directory = os.path.dirname(os.path.abspath(__file__))
		devset, testset, trainset = self.datasets_from_tfrecords(
			script_directory,
			dataset_directory="data/IIGS6Lvl10/tfrecords/tfrecord_dataset_IIGS6_1_6_false_true_lvl10"
		)
		return devset, testset, trainset

	def train_one_epoch(self, trainset):
		for sample in trainset.next_batch(self.data_session):
			print("\tTraining batch #{}".format(trainset.batch_id))
			reaches, targets = sample
			self.network.train(reaches, targets)

	def evaluate_devset(self, devset, epoch):
		for sample in devset.next_batch(self.data_session):
			reaches, targets = sample
			devset_error_mse, devset_error_infinity = self.network.evaluate("dev", reaches, targets)
			print("[epoch #{}, dev batch #{}] dev MSE {}, \tdev L-infinity error {}".format(
				epoch, devset.batch_id, devset_error_mse, devset_error_infinity))

	def evaluate_testset(self, testset):
		for sample in testset.next_batch(self.data_session):
			reaches, targets = sample
			testset_error_mse, testset_error_infinity = self.network.evaluate("test", reaches, targets)
			print("[test batch #{}] test MSE {}, \ttest L-infinity error {}".format(
				testset.batch_id, testset_error_mse, testset_error_infinity))

	def showcase_predictions(self, trainset):
		pass

	def run_neural_net(self):
		with tf.Session() as self.data_session:
			super().run_neural_net()


# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = True


if __name__ == '__main__' and ACTIVATE_FILE:
	runner = Runner_CNN_IIGS6Lvl10_TFRecords()
	runner.run_neural_net()
