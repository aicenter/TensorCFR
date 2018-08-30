#!/usr/bin/env python3
import datetime
import os
import subprocess


def get_current_timestamp():
	return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


# def get_memory_usage():
# 	process = psutil.Process(os.getpid())
# 	return process.memory_info().rss


def get_memory_usage():
	pid = os.getpid()
	cmd = ['ps', '-q', str(pid), '-o', 'rss']

	proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	(out, err) = proc.communicate()

	if len(err) == 0:
		out_list = out.decode('ASCII').strip().split('\n')
		return int(out_list[1])
	else:
		return None