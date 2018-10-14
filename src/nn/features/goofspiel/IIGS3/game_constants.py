#!/usr/bin/env python3

FEATURES_BASENAME = "IIGS3_1_3_false_true_lvl7"
N_CARDS = 3
N_ROUNDS = 2
NAMES_OF_FEATURE_CSV = [
	"private_card1", "private_card2",
	"opponent_card1", "opponent_card2",
	"round1", "round2"
]
FEATURE_COLUMNS = [
	"round1", "round2",
	"private_card1", "private_card2",
	"opponent_card1", "opponent_card2",
	"nodal_reach"
]
TARGET_COLUMNS = ["nodal_expected_value"]
SLICE_1HOT_FEATS = slice(6)
