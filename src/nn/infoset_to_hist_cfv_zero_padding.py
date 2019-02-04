import pandas as pd
import numpy as np
import os
from src.algorithms.tensorcfr_nn import TensorCFR_NN
from src.nn.data.preprocessing_ranges import *
from src.nn.AbstractNN import AbstractNN

nn = AbstractNN()

inf_to_hist_zero = np.expand_dims(np.zeros(120**2),axis=1)







def infoset_to_hist_cfv_zero_padding(nn_out=None):

	if nn_out.shape != (27,120):
		raise ValueError

	else:
		## this version is only for nns that output cfv of p1. meaning a vector of size 120 for each public state
		##TODO solve string with negative numbers to tuple conversion in a fast way
		import numpy as np
		from src.nn.data.preprocessing_ranges import load_input_mask,load_output_mask,load_history_identifier,filter_by_public_state,filter_by_card_combination

		output = load_output_mask()
		hist_id = load_history_identifier()

		final = np.zeros(120 ** 2)

		public_states = output.index[:27]

		infosets = output.columns[:120]

		indices = hist_id.index

		nn_out.index = public_states
		nn_out.columns =infosets

		for public_state in public_states:

			for infoset in infosets:

				infset_cfv = nn_out.loc[public_state,infoset]

				infset_histories = get_indices_hist_in_infoset(public_state,infoset,1)

				final[infset_histories[0]] = infset_cfv

				final[infset_histories[1:]] = 0
