#!/usr/bin/env python3
import argparse
import datetime
import os
import subprocess

import numpy as np
import pandas as pd


def activate_script():
	enviroment_variable = 'TENSORCFR_DEACTIVATE_LONG_RUNNING_SCRIPTS'
	if enviroment_variable in os.environ and int(os.environ[enviroment_variable]) == 1:
		return False
	return True


def get_current_timestamp():
	return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def get_memory_usage():
	# process = psutil.Process(os.getpid())
	# return process.memory_info().rss
	raise NotImplementedError


def get_memory_usage_using_os():
	pid = os.getpid()
	cmd = ['ps', '-q', str(pid), '-o', 'rss']

	proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
	(out, err) = proc.communicate()

	if len(err) == 0:
		out_list = out.decode('ASCII').strip().split('\n')
		return int(out_list[1])
	else:
		return None


def get_dataset_parameters(domain_name):
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", help="dataset seed", nargs='?', type=int, default=0)
	parser.add_argument("--size", help="size of one dataset (given the seed)", nargs='?', type=int, default=4)
	args = parser.parse_args()
	return {
		"domain_name"  : domain_name,
		"dataset_size" : args.size,
		"starting_seed": args.seed
	}


def print_dataset_parameters(dataset_parameters):
	print("###################################")
	print("domain name: {}".format(dataset_parameters["domain_name"]))
	print("starting seed: {}".format(dataset_parameters["starting_seed"]))
	print("dataset size: {}".format(dataset_parameters["dataset_size"]))
	print("###################################")


def get_files_in_directory_recursively(rootdir):
	filenames = []
	for root, dirs, files in os.walk(rootdir):
		for file in files:
			filenames += [("{}/{}".format(root, file))]
	return filenames


def one_hot(a, num_classes):
	return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def get_one_hot_flattened(features, n_classes, slice_1hot_feats):
	indices = (features[..., slice_1hot_feats]).astype(int)
	one_hot_features = one_hot(indices, n_classes)
	one_hot_shape = list(indices.shape)
	one_hot_shape[-1] *= n_classes
	return one_hot_features.reshape(one_hot_shape)


def get_features_dataframe(filename, names, quiet=False):
	features_dataframe = pd.read_csv(
		filename,
		names=names,
		delimiter=";|,",
	)
	if not quiet:
		print("features:")
		print(features_dataframe)
	return features_dataframe
