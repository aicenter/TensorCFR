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
	pass


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
		from src.nn.data.preprocessing_ranges import load_infoset_hist_ids,load

		tensor_cfr_in = np.zeros(120 ** 2)

		inf_id_p1 = load_infoset_hist_ids().iloc[:,:120]

		info_list = load_infoset_list()

		for id in info_list:

			tensor_cfr_in[id] = nn_out.iloc[where(inf_id_p1 == id)]

		return tensor_cfr_in

