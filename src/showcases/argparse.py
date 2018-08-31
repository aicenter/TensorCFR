#!/usr/bin/env python3
# from https://docs.python.org/3/howto/argparse.html#getting-a-little-more-advanced

import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# positional arguments
	parser.add_argument("x", type=int, help="the base")
	parser.add_argument("y", type=int, help="the exponent")
	# optional arguments
	parser.add_argument("-v", "--verbosity", action="count", default=0)

	args = parser.parse_args()
	answer = args.x**args.y
	if args.verbosity >= 2:
		print("Running '{}'".format(__file__))
	if args.verbosity >= 1:
		print("{}^{} == ".format(args.x, args.y), end="")
	print(answer)
