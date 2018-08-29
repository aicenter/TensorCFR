#!/usr/bin/env python3
import os

import psutil

if __name__ == '__main__':
	"""
	Demonstrate that `lambda x: zip(*x)` is self-inverse. In other words, Python function `zip` is idempotent.
	"""
	process = psutil.Process(os.getpid())
	print("memory: {} bytes".format(process.memory_info().rss))
