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

if __name__ == '__main__' and activate_script():
	domain = get_domain_by_name("IIGS6_gambit_flattened")

	computation_graph = tf.Graph()
	with computation_graph.as_default():
		tensorcfr = TensorCFR_Goofstack(domain,load_nn("300.hdf5"),trunk_depth=0)

		steps_to_register = [0, 1, 2, 3, 4, 5]
		tensorcfr.run_cfr(
			total_steps=6,
			delay=2,
			verbose=True,
			register_strategies_on_step=steps_to_register
		)
		average_strategies_over_steps = tensorcfr.average_strategies_over_steps

	tensorcfr = TensorCFR_Goofstack(domain, load_nn("300.hdf5"), trunk_depth=10)

	tensorcfr.run_cfr(total_steps=10,delay=5)
	#tensorcfr.average_strategies_over_steps
	tensorcfr_expl = ExploitabilityByTensorCFR(domain_numpy=domain,trunk_depth=10,
	                                           trunk_strategies=tensorcfr.average_strategies_over_steps['average_strategy_step9'])

	myexpl= tensorcfr_expl.evaluate(tensorcfr.average_strategies_over_steps['average_strategy_step9'],10)

	best_response = TensorCFR_BestResponse(best_responder=2,trunk_strategies=avgstr,domain=domain,trunk_depth=10)

	#input_reaches = tf.range(len(nn_input_permutation), name="input_reaches")
	equilibrium_values = tensorcfr.predict_equilibrial_values(mydf)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print_tensors(sess, [np.ones(120**2), equilibrium_values])
