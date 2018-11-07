#!/usr/bin/env python3
import logging

import tensorflow as tf

from src.algorithms.tensorcfr_nn.TensorCFR_NN import TensorCFR_NN
from src.domains.available_domains import get_domain_by_name
from src.nn.DenseNet_IIGS3Lvl7 import DenseNet_IIGS3Lvl7
from src.nn.data.DatasetFromNPZ import DatasetFromNPZ
from src.nn.features.goofspiel.IIGS3.sorting_permutation_by_public_states import get_permutation_by_public_states
from src.utils.other_utils import get_current_timestamp
from src.utils.tf_utils import print_tensors

# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = True


def create_logger(log_lvl=logging.WARNING):
	log_filename = "logs/tensorcfr_CNN_IIGS3_td7_{}.log".format(get_current_timestamp())
	logging.basicConfig(filename=log_filename, format='%(asctime)s %(message)s', level=log_lvl)


if __name__ == '__main__' and ACTIVATE_FILE:
	import datetime
	import os
	import re

	create_logger()

	args = DenseNet_IIGS3Lvl7.parse_arguments()
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
	network = DenseNet_IIGS3Lvl7(threads=args.threads)
	network.construct(args)
	network.saver.restore(sess=network.session, save_path="/home/dominik/PycharmProjects/TensorCFR/src/nn/mymodel_final.ckpt")
	print("network restored")
	domain_ = get_domain_by_name("II-GS3_gambit_flattened")
	nn_input_permutation = get_permutation_by_public_states()
	tensorcfr = TensorCFR_NN(
		domain_,
		neural_net=network,
		nn_input_permutation=nn_input_permutation,
		trunk_depth=7
	)

	# Train
	for epoch in range(args.epochs):

		# Evaluate on development set
		devset_error_mse, devset_error_infinity = network.evaluate("dev", devset.features, devset.targets)
		logging.info("[epoch #{}] dev MSE {}, \tdev L-infinity error {}".format(epoch, devset_error_mse, devset_error_infinity))

	# Evaluate on test set
	testset_error_mse, testset_error_infinity = network.evaluate("test", testset.features, testset.targets)
	logging.info("\nmean squared error on testset: {}".format(testset_error_mse))
	logging.info("L-infinity error on testset: {}".format(testset_error_infinity))

	mockup_input_reaches = tf.divide(
		tf.range(len(nn_input_permutation)),
		1000,
		name="mockup_input_reaches"
	)
	mockup_equilibrium_values = tensorcfr.predict_equilibrial_values(
		mockup_input_reaches,
		name="mockup_equilibrium_values"
	)
	equilibrium_values = tensorcfr.predict_equilibrial_values()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print_tensors(sess, [mockup_input_reaches, mockup_equilibrium_values, equilibrium_values])

	steps_to_register = [0, 2, 4]
	tensorcfr.run_cfr(total_steps=5, delay=2, verbose=True, register_strategies_on_step=steps_to_register)
	#
	# for step in steps_to_register:
	# 	trunk_strategy = tensorcfr.average_strategies_over_steps["average_strategy_step{}".format(step)]
	# 	print("step {}:".format(step))
	# 	logging.info("average_strategy_step{}:\n{}".format(step, trunk_strategy))
	#
	# 	exploitability_tensorcfr = ExploitabilityByTensorCFR(
	# 		domain_,
	# 		trunk_depth=7,
	# 		trunk_strategies=trunk_strategy,
	# 		total_steps=10,
	# 		delay=3,
	# 		log_lvl=logging.INFO
	# 	)
	# 	logging.info("BR value (player 1) at step {}: {}".format(step, exploitability_tensorcfr.final_brvalue_1))
	# 	logging.info("BR value (player 2) at step {}: {}".format(step, exploitability_tensorcfr.final_brvalue_2))
	# 	print("final exploitability: {}".format(exploitability_tensorcfr.final_exploitability))
