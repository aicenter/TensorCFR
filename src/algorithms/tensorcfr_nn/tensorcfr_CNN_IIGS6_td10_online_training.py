#!/usr/bin/env python3
import logging

from src.algorithms.tensorcfr_best_response.ExploitabilityByTensorCFR import ExploitabilityByTensorCFR
from src.algorithms.tensorcfr_nn.TensorCFR_NN import TensorCFR_NN
from src.domains.available_domains import get_domain_by_name
from src.nn.Runner_CNN_IIGS6Lvl10_NPZ import Runner_CNN_IIGS6Lvl10_NPZ
from src.nn.features.goofspiel.IIGS6.sorting_permutation_by_public_states import get_permutation_by_public_states
from src.utils.other_utils import get_current_timestamp

# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = False


def create_logger(log_lvl=logging.WARNING):
	log_filename = "logs/tensorcfr_CNN_IIGS6_td10_{}.log".format(get_current_timestamp())
	logging.basicConfig(filename=log_filename, format='%(asctime)s %(message)s', level=log_lvl)


if __name__ == '__main__' and ACTIVATE_FILE:
	create_logger()

	runner = Runner_CNN_IIGS6Lvl10_NPZ()
	runner.run_neural_net()

	network = runner.network
	domain_ = get_domain_by_name("IIGS6_gambit_flattened")
	nn_input_permutation = get_permutation_by_public_states()
	tensorcfr = TensorCFR_NN(
		domain_,
		neural_net=network,
		nn_input_permutation=nn_input_permutation,
		trunk_depth=10
	)

	steps_to_register = [0, 1, 2, 3, 4, 5]
	tensorcfr.run_cfr(
		total_steps=6,
		delay=2,
		verbose=True,
		register_strategies_on_step=steps_to_register
	)
	average_strategies_over_steps = tensorcfr.average_strategies_over_steps
	del tensorcfr

	for step in steps_to_register:
		trunk_strategy = average_strategies_over_steps["average_strategy_step{}".format(step)]
		logging.info("average_strategy_step{}:\n{}".format(step, trunk_strategy))

		exploitability_tensorcfr = ExploitabilityByTensorCFR(   # TODO optimize by construction object only once
			domain_,
			trunk_depth=10,
			trunk_strategies=trunk_strategy,
			total_steps=100,
			delay=25,
			log_lvl=logging.INFO
		)
		logging.debug("BR value (player 1) at step {}: {}".format(step, exploitability_tensorcfr.final_brvalue_1))
		logging.debug("BR value (player 2) at step {}: {}".format(step, exploitability_tensorcfr.final_brvalue_2))
		logging.info(
			"exploitability of avg strategy at step {}: {}".format(step, exploitability_tensorcfr.final_exploitability))
		del exploitability_tensorcfr
