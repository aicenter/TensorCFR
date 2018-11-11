#!/usr/bin/env python3
import logging

import tensorflow as tf

from src.algorithms.tensorcfr_best_response.ExploitabilityByTensorCFR import ExploitabilityByTensorCFR
from src.algorithms.tensorcfr_nn.TensorCFR_NN import TensorCFR_NN
from src.commons.constants import PROJECT_ROOT
from src.domains.FlattenedDomain import FlattenedDomain
from src.nn.Runner_CNN_IIGS6Lvl10_TFRecords import Runner_CNN_IIGS6Lvl10_TFRecords
from src.nn.features.goofspiel.IIGS6.sorting_permutation_by_public_states import get_permutation_by_public_states
from src.utils.gambit_flattened_domains.loader import GambitLoaderCached
from src.utils.other_utils import get_current_timestamp

# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = False


def create_logger(log_lvl=logging.WARNING, log_to_file=True):
	fmt = '%(asctime)s %(message)s'
	if log_to_file:
		logging.basicConfig(
			filename="logs/tensorcfr_CNN_IIGS6_td10_{}.log".format(get_current_timestamp()),
			format=fmt,
			level=log_lvl
		)
	else:
		logging.basicConfig(
			format=fmt,
			level=log_lvl
		)


if __name__ == '__main__' and ACTIVATE_FILE:
	import os

	create_logger(
		log_lvl=logging.INFO,
		log_to_file=False
	)

	runner = Runner_CNN_IIGS6Lvl10_TFRecords()
	ckpt_dir = "checkpoints/Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_021836-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,r=C-138," \
	           "t=1,tr=0.8"
	ckpt_basename = "final_2018-11-11_07:46:15.ckpt"
	runner.restore_from_ckpt(ckpt_dir, ckpt_basename)
	network = runner.network

	steps_to_register = list()
	average_strategies_over_steps = dict()

	path_to_domain_efg = os.path.join(
		PROJECT_ROOT,
		'doc',
		'goofspiel',
		'IIGS6_s1_bf_ft.gbt'
	)

	domain_in_numpy = GambitLoaderCached(path_to_domain_efg)

	computation_graph = tf.Graph()
	with computation_graph.as_default():
		domain = FlattenedDomain(
			"IIGS6",
			domain_in_numpy.domain_parameters,
			domain_in_numpy.number_of_nodes_actions,
			domain_in_numpy.node_to_infoset,
			domain_in_numpy.utilities,
			domain_in_numpy.infoset_acting_players,
			domain_in_numpy.initial_infoset_strategies,
			information_set_mapping_to_gtlibrary=domain_in_numpy.information_set_mapping_to_gtlibrary
		)

		nn_input_permutation = get_permutation_by_public_states()
		tensorcfr = TensorCFR_NN(
			domain,
			neural_net=network,
			nn_input_permutation=nn_input_permutation,
			trunk_depth=10
		)

		steps_to_register = [0, 200, 400, 600, 800, 999]
		tensorcfr.run_cfr(
			total_steps=1000,
			delay=250,
			verbose=True,
			register_strategies_on_step=steps_to_register
		)
		average_strategies_over_steps = tensorcfr.average_strategies_over_steps
	del computation_graph
	del tensorcfr

	exploitability_tensorcfr = ExploitabilityByTensorCFR(
		domain_in_numpy,
		trunk_depth=None,
		trunk_strategies=None,
		total_steps=10000,
		delay=2500,
		log_lvl=logging.INFO
	)

	exploitabilities = {}
	br_values1 = {}
	br_values2 = {}
	for step in steps_to_register:
		print("\n########## CFR step {}: exploitability evaluation ##########".format(step))
		trunk_strategy = average_strategies_over_steps["average_strategy_step{}".format(step)]
		logging.debug("average_strategy_step{}:\n{}".format(step, trunk_strategy))

		exploitability_tensorcfr.evaluate(trunk_strategies=trunk_strategy, trunk_depth=10)
		br_values1[step] = exploitability_tensorcfr.final_brvalue_1
		br_values2[step] = exploitability_tensorcfr.final_brvalue_2
		exploitabilities[step] = exploitability_tensorcfr.final_exploitability

	for step in steps_to_register:
		logging.debug("BR value (player 1) at step {}: {}".format(step, br_values1[step]))
		logging.debug("BR value (player 2) at step {}: {}".format(step, br_values2[step]))
		print("exploitability of avg strategy at step {}: {}".format(step, exploitabilities[step]))
