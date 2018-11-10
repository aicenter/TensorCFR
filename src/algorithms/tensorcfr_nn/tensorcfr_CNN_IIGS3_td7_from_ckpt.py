#!/usr/bin/env python3
import logging

from src.nn.ConvNet_IIGS3Lvl7 import ConvNet_IIGS3Lvl7
from src.nn.data.DatasetFromNPZ import DatasetFromNPZ
from src.utils.other_utils import get_current_timestamp

# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = False


def create_logger(log_lvl=logging.WARNING):
	logging.basicConfig(format='%(asctime)s %(message)s', level=log_lvl)


if __name__ == '__main__' and ACTIVATE_FILE:
	import datetime
	import os
	import re

	create_logger(
		log_lvl=logging.INFO
	)

	args = ConvNet_IIGS3Lvl7.parse_arguments()
	print("args: {}".format(args))
	# Create logdir name
	args.logdir = "logs/{}-{}-{}".format(
	os.path.basename(__file__),
		datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
		",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
	)
	if not os.path.exists("logs"):
		os.mkdir("logs")  # TF 1.6 will do this by itself

	# Load the data
	script_directory = os.path.dirname(os.path.abspath(__file__))
	dataset_directory = "../../nn/data/IIGS3Lvl7/80-10-10_only_reaches"
	npz_basename = "IIGS3_1_3_false_true_lvl7"
	trainset = DatasetFromNPZ("{}/{}/{}_train.npz".format(script_directory, dataset_directory, npz_basename))
	devset = DatasetFromNPZ("{}/{}/{}_dev.npz".format(script_directory, dataset_directory, npz_basename))
	testset = DatasetFromNPZ("{}/{}/{}_test.npz".format(script_directory, dataset_directory, npz_basename))

	# Construct the network
	network = ConvNet_IIGS3Lvl7(threads=args.threads)
	network.construct(args)
	# Train
	for epoch in range(args.epochs):
		while not trainset.epoch_finished():
			reaches, targets = trainset.next_batch(args.batch_size)
			network.train(reaches, targets)
		# Evaluate on development set
		devset_error_mse, devset_error_infinity = network.evaluate("dev", devset.features, devset.targets)
		logging.info("[epoch #{}] dev MSE {}, \tdev L-infinity error {}".format(epoch, devset_error_mse, devset_error_infinity))
	# Evaluate on test set
	testset_error_mse, testset_error_infinity = network.evaluate("test", testset.features, testset.targets)
	logging.info("mean squared error on testset: {}".format(testset_error_mse))
	logging.info("L-infinity error on testset: {}".format(testset_error_infinity))

	ckpt_dir = "checkpoints"
	ckpt_basename = "tensorcfr_CNN_IIGS3_td7_{}".format(get_current_timestamp())
	network.save_to_ckpt(ckpt_dir, ckpt_basename)
	del network

	restored_network = ConvNet_IIGS3Lvl7(threads=args.threads)
	restored_network.construct(args)
	restored_network.restore_from_ckpt(ckpt_dir, ckpt_basename)

	testset_error_mse, testset_error_infinity = restored_network.evaluate("test", testset.features, testset.targets)
	logging.info("[restored] mean squared error on testset: {}".format(testset_error_mse))
	logging.info("[restored] L-infinity error on testset: {}".format(testset_error_infinity))
