#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py
import tensorflow as tf
import numpy as np
from src.commons.constants import SEED_FOR_TESTING
from src.nn.Runner_CNN_Selu_InfinityLoss_IIGS6Lvl10_NPZ import Runner_CNN_Selu_InfinityLoss_IIGS6Lvl10_NPZ
from src.nn.data.DatasetFromTFRecord import DatasetFromTFRecord
from src.nn.features.goofspiel.IIGS6.tfrecords_reaches_values_IIGS6_1_6_false_true_lvl10 import FEATURES_PER_FILE
from src.utils.other_utils import get_files_in_directory_recursively
from src.nn.ConvNet_Selu_InfinityLoss_IIGS6Lvl10 import ConvNet_Selu_InfinityLoss_IIGS6Lvl10

class Runner_CNN_Selu_InfinityLoss_IIGS6Lvl10_TFRecords(Runner_CNN_Selu_InfinityLoss_IIGS6Lvl10_NPZ):
	def __init__(self, fixed_randomness=False):
		super().__init__(fixed_randomness)
		self.data_session = None

	@property
	def default_extractor_arch(self):
		return "C-{}".format(ConvNet_Selu_InfinityLoss_IIGS6Lvl10.INPUT_FEATURES_DIM)

	@property
	def default_regressor_arch(self):
		return "C-{}".format(ConvNet_Selu_InfinityLoss_IIGS6Lvl10.INPUT_FEATURES_DIM)

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

		SAMPLE_LENGTH = 14400
		trainset = DatasetFromTFRecord(
			batch_size=self.args.batch_size,
			dataset_files=trainset_files,
			sample_length=SAMPLE_LENGTH,
			shuffle_batches=True,
			variable_scope_name='trainset'
		)
		devset = DatasetFromTFRecord(
			batch_size=self.args.batch_size,
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
		pass

	# def run_neural_net(self):
	# 	with tf.Session() as self.data_session:
	# 		super().run_neural_net()

	def construct_network(self):
		network = ConvNet_Selu_InfinityLoss_IIGS6Lvl10(threads=self.args.threads, fixed_randomness=self.fixed_randomness)
		network.construct(self.args)
		return network

	def run_neural_net(self):
		with tf.Session() as self.data_session:
			np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)
			if self.fixed_randomness:
				print("Abstract: self.fixed_randomness is {}".format(self.fixed_randomness))
				np.random.seed(SEED_FOR_TESTING)  # Fix random seed

			self.parse_arguments()
			dataset_directory = self.args.dataset_directory
			self.create_logdir()

			devset, testset, trainset = self.init_datasets(dataset_directory)
			self.network = self.construct_network()

			for self.epoch in range(self.args.epochs):
				self.train_one_epoch(trainset)
				self.evaluate_devset(devset)

			self.evaluate_testset(testset)
			self.showcase_predictions(trainset)


# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = False

if __name__ == '__main__' and ACTIVATE_FILE:
	runner = Runner_CNN_Selu_InfinityLoss_IIGS6Lvl10_TFRecords()
	runner.run_neural_net()
