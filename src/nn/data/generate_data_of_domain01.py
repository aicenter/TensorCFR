#!/usr/bin/env python3
import os

from src.nn.data.generate_data import generate_data

if __name__ == '__main__':
	generate_data(
		"flattened_domain01_via_gambit",
		script_directory=os.path.dirname(os.path.abspath(__file__)),
		trunk_depth=2
	)
