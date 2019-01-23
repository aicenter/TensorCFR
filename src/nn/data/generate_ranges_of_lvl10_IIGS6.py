#!/usr/bin/env python3
import os

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.domains.available_domains import get_domain_by_name
from src.nn.data.generate_data import generate_data
from src.utils.other_utils import activate_script
import tensorflow as tf

trunk_depth = 10
domain = get_domain_by_name("IIGS6_gambit_flattened")
tensorcfr = TensorCFRFixedTrunkStrategies(domain)


session = tf.Session(graph=tensorcfr)

with session.as_default():
	tf.Print(example)

if __name__ == '__main__' and activate_script():
	generate_data(
		"IIGS6_gambit_flattened",
		script_directory=os.path.dirname(os.path.abspath(__file__)),
		trunk_depth=10
	)
