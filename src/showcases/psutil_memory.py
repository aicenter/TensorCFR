#!/usr/bin/env python3

from src.utils.other_utils import get_memory_usage

if __name__ == '__main__':
	"""
	Demonstrate that `lambda x: zip(*x)` is self-inverse. In other words, Python function `zip` is idempotent.
	"""
	print("memory: {:,} bytes".format(get_memory_usage()))
