#!/usr/bin/env python3
import os

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.nn.data.generate_data import generate_data

# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = True


if __name__ == '__main__' and ACTIVATE_FILE:
	generate_data(
		"II-GS2_gambit_flattened",
		script_directory=os.path.dirname(os.path.abspath(__file__)),
		trunk_depth=4
	)
