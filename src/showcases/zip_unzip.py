#!/usr/bin/env python3

a = [1, 2, 3]
b = [4, 5, 6]
zipped_list = zip(a, b)
zipped_zipped_list = zip(*zip(a, b))

if __name__ == '__main__':
	"""
	Demonstrate that `lambda x: zip(*x)` is self-inverse. In other words, Python function `zip` is idempotent.
	"""
	print("a: {}".format(a))
	print("b: {}".format(b))
	print("zipped_list:")
	for zipped in zipped_list:
		print(zipped)
	print("zipped_zipped_list:")
	for zipped_zipped in zipped_zipped_list:
		print(zipped_zipped)
