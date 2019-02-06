import pandas as pd
import numpy as np
import os
from src.algorithms.tensorcfr_nn import TensorCFR_NN
from src.nn.data.preprocessing_ranges import *
from src.nn.AbstractNN import AbstractNN

nn = AbstractNN()

inf_to_hist_zero = np.expand_dims(np.zeros(120**2),axis=1)


def tensorcfr_to_nn_input(tensor_cfr_out=None):

	##TODO get ranges from tensorcfrfixestrunk. bring them into format [public_state,ranges p1] for each publicstate
	## TODO implement range of ifnoset in tensorcfr. its easier

	mask = load_input_mask()
	hist_id = load_history_identifier()
	public_states_list = [(x,y,z) for x in [0, 1, -1] for y in [0, 1, -1] for z in [0, 1, -1]]

	for public_state in public_states_list:

		df_by_public_state = filter_by_public_state(hist_id,public_state)


		for cards in mask.columns[3:123]:
			## for player 1

			cards_df = filter_by_card_combination(df_by_public_state,cards,1)

			if cards_df.shape[0] >= 1:

				# puts range of p1 in of infoset "cards" of public state "public_state" into mask

				mask.loc["".join(tuple(map(str,public_state))),cards] = float(tensor_cfr_out.iloc[tensor_cfr_out.index == cards_df.index[0],0])

			else:

				mask.loc["".join(tuple(map(str,public_state))), cards] = 0


		for cards in mask.columns[123:]:

			cards_df = filter_by_card_combination(df_by_public_state, cards, 2)

			if cards_df.shape[0] == 1:

				mask.loc["".join(tuple(map(str,public_state))), cards] = float(tensor_cfr_out.iloc[tensor_cfr_out.index == cards_df.index[0], 1])



			elif cards_df.shape[0] > 1:

				mask.loc["".join(tuple(map(str,public_state))), cards] = float(tensor_cfr_out.iloc[tensor_cfr_out.index == cards_df.index[0], 1])


			elif cards_df.shape[0] == 0:

				mask.loc["".join(tuple(map(str,public_state))), cards] = 0



	return mask


def load_nn(path_to_ckpt=None):
	## TODO load nn from hdf5 ckpt
	pass

def nn_out_to_tensorcfr_in(nn_out=None):

	if nn_out.shape != (27,120):
		raise ValueError

	else:
		## this version is only for nns that output cfv of p1. meaning a vector of size 120 for each public state
		##TODO solve string with negative numbers to tuple conversion in a fast way
		from numpy import where
		from src.nn.data.preprocessing_ranges import load_infoset_hist_ids,load_infoset_list

		tensor_cfr_in = np.zeros(120 ** 2)

		inf_id_p1 = load_infoset_hist_ids().iloc[:,:120]

		info_list = load_infoset_list()

		for id in info_list:

			tensor_cfr_in[id] = nn_out.iloc[where(inf_id_p1 == id)]

		return tensor_cfr_in

