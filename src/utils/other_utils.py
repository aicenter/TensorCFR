#!/usr/bin/env python3
import datetime
import os
import subprocess

import psutil


def get_current_timestamp():
	return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def get_memory_usage():
	process = psutil.Process(os.getpid())
	return process.memory_info().rss


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


def print_dataset_parameters(domain_name, starting_seed, dataset_size):
	print("###################################")
	print("domain name: {}".format(domain_name))
	print("starting seed: {}".format(starting_seed))
	print("dataset size: {}".format(dataset_size))
	print("###################################")