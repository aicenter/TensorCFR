#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py
from src.nn.ConvNet_IIGS6Lvl10 import ConvNet_IIGS6Lvl10


class FullyConvNet_IIGS6Lvl10_no_context_pooling(ConvNet_IIGS6Lvl10):
	def construct_context_pooling(self):
		print("Context pooling skipped for {}".format(self.__class__.__name__))
