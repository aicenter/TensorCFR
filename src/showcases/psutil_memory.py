#!/usr/bin/env python3

from src.utils.other_utils import get_memory_usage

if __name__ == '__main__':
	print("memory: {:,} bytes".format(get_memory_usage()))
