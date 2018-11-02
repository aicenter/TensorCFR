#!/usr/bin/env python3
import tensorflow as tf

from src.algorithms.tensorcfr_nn.TensorCFR_NN import TensorCFR_NN
from src.domains.available_domains import get_domain_by_name
from src.nn.ConvNet_IIGS3Lvl7 import ConvNet_IIGS3Lvl7
from src.nn.features.goofspiel.IIGS3.sorting_permutation_by_public_states import get_permutation_by_public_states
from src.utils.tf_utils import print_tensors

if __name__ == '__main__':
	import datetime
	import os
	import re

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

	domain_ = get_domain_by_name("II-GS3_gambit_flattened")
	network = ConvNet_IIGS3Lvl7(threads=None)
	network.construct(args)
	nn_input_permutation = get_permutation_by_public_states()
	tensorcfr = TensorCFR_NN(
		domain_,
		neural_net=network,
		nn_input_permutation=nn_input_permutation,
		trunk_depth=7
	)

	input_reaches = tf.range(len(nn_input_permutation), name="input_reaches") / 1000
	equilibrium_values = tensorcfr.predict_equilibrial_values(input_reaches)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print_tensors(sess, [input_reaches, equilibrium_values])
