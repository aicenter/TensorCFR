#!/usr/bin/env python3
import os

from src.nn.data.generate_data import generate_data
from src.utils.other_utils import activate_script


if __name__ == '__main__' and activate_script():
	generate_data(
		"GP_cards4x3_222_gambit_flattened",
		script_directory=os.path.dirname(os.path.abspath(__file__)),
		trunk_depth=6
	)
