#!/usr/bin/env python3

import os

from src.utils.other_utils import get_files_in_directory_recursively

if __name__ == '__main__':
	script_directory = os.path.dirname(os.path.abspath(__file__))
	rootdir = "{}/reach_value_datasets".format(script_directory)

	filenames = get_files_in_directory_recursively(rootdir)
	for filename in filenames:
		print(filename)
	print("{} files".format(len(filenames)))
