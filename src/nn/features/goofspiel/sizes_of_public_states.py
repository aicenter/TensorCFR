#!/usr/bin/env python3

import pandas as pd

if __name__ == '__main__':
	features = pd.read_csv("IIGS3_features.csv", names=["private_card1", "private_card2", "round1", "round2"])
	# features = pd.read_csv("IIGS3_1_3_false_true_lvl7.csv")
	print(features.head())
