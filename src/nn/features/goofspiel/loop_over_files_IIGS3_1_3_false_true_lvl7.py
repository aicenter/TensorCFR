#!/usr/bin/env python3

import os


if __name__ == '__main__':
	script_directory = os.path.dirname(os.path.abspath(__file__))
	rootdir = "{}/reach_value_datasets".format(script_directory)

	filenames = []
	for root, dirs, files in os.walk(rootdir):
		for file in files:
			filenames += [("{}/{}".format(root, file))]

	for filename in filenames:
		print(filename)
	print("{} files".format(len(filenames)))
