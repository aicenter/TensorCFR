#!/usr/bin/env python3
import tensorflow as tf

from src.algorithms.tensorcfr_nn.TensorCFR_Goofstack import TensorCFR_Goofstack
from src.algorithms.tensorcfr_best_response.ExploitabilityByTensorCFR import ExploitabilityByTensorCFR
from src.algorithms.tensorcfr_best_response.TensorCFR_BestResponse import TensorCFR_BestResponse
from src.domains.available_domains import get_domain_by_name
from src.nn.data.postprocessing_ranges import load_nn
from src.utils.other_utils import activate_script
from src.utils.tf_utils import print_tensors
import numpy as np
from src.nn.data.preprocessing_ranges import get_files_in_directory_recursively
import os
import pickle

# with open("/home/dominik/PycharmProjects/TensorCFR/src/algorithms/"+"tensorcfr_best_responseexpl_per_epoch.pickle","rb") as f:
# 	mydict = pickle.load(f)

#if __name__ == '__main__' and activate_script():


## TODO please run the same network 10 times for 10 iterations of CFR. Compute exploitability and compare numbers
domain = get_domain_by_name("IIGS6_gambit_flattened")

if __name__ == '__main__' and activate_script():
	nn = load_nn("your path to TensorCFR"+"/TensorCFR/experiments/Goofstack_Experiments/non_determinism_sanity_check/200.hdf5")

	mysteps = 10

	expl_dict = {}


	for run in range(10):

		print("expl for {}".format(run))

		tensorcfr = TensorCFR_Goofstack(domain,nn,trunk_depth=10)

		tensorcfr.run_cfr(total_steps=mysteps, delay=5)

		print("tensorcfr iterations done..")

		final_expl= []

		for player in [1,2]:

			print("bestresponse for player {}".format(player))

			best_response = TensorCFR_BestResponse(best_responder=player,trunk_strategies=tensorcfr.average_strategies_over_steps['average_strategy_step'+str(mysteps-1)],
		                                       domain=domain,trunk_depth=10)

			final_expl.append(best_response.get_final_best_response_value(total_steps=1))

		myexpl = (np.abs(final_expl[0])+np.abs(final_expl[1]))/2

		print("{} expl is: {}".format(run,myexpl))
		expl_dict["{}".format(run)] = myexpl

	for key,value in expl_dict:
		print(key + "run has expl " +str(value))