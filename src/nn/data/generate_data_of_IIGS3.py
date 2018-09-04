#!/usr/bin/env python3
import os

from src.nn.data.generate_data import generate_data

if __name__ == '__main__':
	generate_data(
		"II-GS3_gambit_flattened",
		script_directory=os.path.dirname(os.path.abspath(__file__)),
		trunk_depth=4
	)
