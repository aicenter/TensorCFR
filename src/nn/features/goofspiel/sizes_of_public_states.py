#!/usr/bin/env python3

import pandas as pd

if __name__ == '__main__':
	data = pd.read_csv("IIGS3_features.csv")
	# data = pd.read_csv("IIGS3_1_3_false_true_lvl7.csv")
	print(data.head())
