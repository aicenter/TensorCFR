#!/usr/bin/env python3
import datetime
import os

import psutil


def get_current_timestamp():
	return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def get_memory_usage():
	process = psutil.Process(os.getpid())
	return process.memory_info().rss
