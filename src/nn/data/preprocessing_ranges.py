import pandas as pd
import numpy as np
import tensorflow as tf
from itertools import permutations
import os


## prepare input format for neural network: [public state,ranges player1, ranges player] for each public state

public_states_list = ["{}{}{}".format(x,y,z) for x in [0,1,-1] for y in [0,1,-1] for z in [0,1,-1]]

#card_combinations_list = [[a,b,c] for a in range(0,6) for b in range(0,6)for c in range(0,6)]

card_combinations_list = list(permutations(range(0,6),3))

empty_input_matrix = pd.DataFrame(index=public_states_list,columns=["r1","r2","r3"]+card_combinations_list+card_combinations_list)

empty_input_matrix.iloc[:,0:3] = np.vstack([np.array([x,y,z]) for x in [0,1,-1] for y in [0,1,-1] for z in [0,1,-1]])

#empty_input_matrix.to_csv(path_or_buf="/home/dominik/PycharmProjects/TensorCFR/src/nn/data/input_mask.csv")

#test = pd.read_csv("/home/dominik/PycharmProjects/TensorCFR/src/nn/data/input_mask.csv",index_col=0)

dat = pd.read_csv("/home/dominik/PycharmProjects/TensorCFR/src/nn/data/out/IIGS6_gambit_flattened/4_datasets/IIGS6_s1_bf_ft_gambit_flattened-2019-01-25_161617-ad=250,ts=1000,td=10/nodal_dataset_seed_0.csv",index_col=False)

#hist = pd.read_csv("/home/dominik/PycharmProjects/TensorCFR/src/nn/features/goofspiel/IIGS6/IIGS6_1_6_false_true_lvl10.csv",names=["r1c1","r1c2","r2c1","r2c2","r3c1","r3c2","r1","r2","r3"])

#test3 = pd.read_csv("/home/dominik/PycharmProjects/TensorCFR/src/nn/data/history_identifier.csv",index_col=0)
def get_files_in_directory_recursively(rootdir):
	filenames = []
	for root, dirs, files in os.walk(rootdir):
		for file in files:
			filenames += [("{}/{}".format(root, file))]
	return filenames

def load_input_mask():
	from pandas import read_csv
	import os
	return read_csv(os.getcwd()+"/src/nn/data/input_mask.csv",index_col=0)

def load_output_mask():
	from pandas import read_csv
	import os
	return read_csv(os.getcwd()+"/src/nn/data/output_mask.csv",index_col=0)

def load_history_identifier():
	from pandas import read_csv
	import os
	return read_csv(os.getcwd()+"/src/nn/data/history_identifier.csv",index_col=0)

def load_seed_from_filepath(path=""):
	return pd.read_csv(path,index_col=False)

def filter_by_public_state(df=None,public_state=""):
	if df is not None:
		df = df.copy()

		if public_state.__len__() == 3:
			return df.loc[(df["r1"]==int(public_state[0])) & (df["r2"]== int(public_state[1])) & (df["r3"]==int(public_state[2]))]
	else:
		return ValueError

def filter_by_card_combination(df=None,cards=None,player=None):

	if player == 1:
		return df.loc[(df["r1c1"]==int(cards[1])) & (df["r2c1"]== int(cards[4])) & (df["r3c1"]==int(cards[7]))]

	if player == 2:
		return df.loc[(df["r1c2"]==int(cards[1])) & (df["r2c2"]== int(cards[4])) & (df["r3c2"]==int(cards[7]))]

def seed_to_sum_cfv_per_infoset(df=None):
	#TODO
	pass

def seed_to_ranges_per_public_state(df=None):
	mask = load_input_mask()
	hist_id = load_history_identifier()
	public_states_list = [(x,y,z) for x in [0, 1, -1] for y in [0, 1, -1] for z in [0, 1, -1]]
	for public_state in public_states_list:

		df_by_public_state = filter_by_public_state(hist_id,public_state)
		print(public_state)
		print(df_by_public_state.shape)

		#for player in [1,2]:

		for cards in mask.columns[3:122]:
			## for player 1

			cards_df = filter_by_card_combination(df_by_public_state,cards,1)



			if cards_df.shape[0] == 1:

				mask.loc["".join(tuple(map(str,public_state))),cards] = float(df.iloc[df.index==cards_df.index[0],3])

			elif cards_df.shape[0] > 1:

				mask.loc["".join(tuple(map(str,public_state))), cards] = float(df.iloc[df.index == cards_df.index[0], 3])

			elif cards_df.shape[0] == 0:

				mask.loc["".join(tuple(map(str,public_state))), cards] = 0

		for cards in mask.columns[123:]:

			cards_df = filter_by_card_combination(df_by_public_state, cards, 2)

			if cards_df.shape[0] == 1:

				mask.loc["".join(tuple(map(str,public_state))), cards] = float(df.iloc[df.index == cards_df.index[0], 4])


			elif cards_df.shape[0] > 1:

				mask.loc["".join(tuple(map(str,public_state))), cards] = float(df.iloc[df.index == cards_df.index[0], 4])

			elif cards_df.shape[0] == 0:

				mask.loc["".join(tuple(map(str,public_state))), cards] = 0

	return mask














def build_training_data(data_dir=""):
	file_list = get_files_in_directory_recursively(data_dir)
	seed_list = [load_seed_from_filepath(seed) for seed in file_list]

	## x

	x = np.vstack([seed_to_ranges_per_public_state(seed)[0] for seed in seed_list])
	y = np.vstack([seed_to_sum_cfv_per_infoset(seed)[1] for seed in seed_list])

	return x,y
