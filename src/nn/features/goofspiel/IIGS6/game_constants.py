#!/usr/bin/env python3

FEATURES_BASENAME = "IIGS6_1_6_false_true_lvl10"
N_CARDS = 6
NAMES_OF_FEATURE_CSV = [
	"private_card1", "private_card2", "private_card3",
	"opponent_card1", "opponent_card2", "opponent_card3",
	"round1", "round2", "round3"
]
FEATURE_COLUMNS = [
	"round1", "round2", "round3"
	"private_card1", "private_card2", "private_card3",
	"opponent_card1", "opponent_card2", "opponent_card3",
	"nodal_reach"
]
TARGET_COLUMNS = ["nodal_expected_value"]
SLICE_1HOT_FEATS = slice(9)
