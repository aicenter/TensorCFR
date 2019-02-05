import pandas as pd
import numpy as np
import tensorflow as tf
from itertools import permutations
import os


## prepare input format for neural network: [public state,ranges player1, ranges player] for each public state

#public_states_list = ["{}{}{}".format(x,y,z) for x in [0,1,-1] for y in [0,1,-1] for z in [0,1,-1]]

#card_combinations_list = [[a,b,c] for a in range(0,6) for b in range(0,6)for c in range(0,6)]

#card_combinations_list = list(permutations(range(0,6),3))

#empty_input_matrix = pd.DataFrame(index=public_states_list,columns=["r1","r2","r3"]+card_combinations_list+card_combinations_list)

#empty_input_matrix.iloc[:,0:3] = np.vstack([np.array([x,y,z]) for x in [0,1,-1] for y in [0,1,-1] for z in [0,1,-1]])

#empty_input_matrix.to_csv(path_or_buf="/home/dominik/PycharmProjects/TensorCFR/src/nn/data/input_mask.csv")

#test = pd.read_csv("/home/dominik/PycharmProjects/TensorCFR/src/nn/data/input_mask.csv",index_col=0)

#dat = pd.read_csv("/home/dominik/PycharmProjects/TensorCFR/src/nn/data/out/IIGS6_gambit_flattened/4_datasets/IIGS6_s1_bf_ft_gambit_flattened-2019-01-25_161617-ad=250,ts=1000,td=10/nodal_dataset_seed_0.csv",index_col=0)

#hist = pd.read_csv("/home/dominik/PycharmProjects/TensorCFR/src/nn/features/goofspiel/IIGS6/IIGS6_1_6_false_true_lvl10.csv",names=["r1c1","r1c2","r2c1","r2c2","r3c1","r3c2","r1","r2","r3"])

#test3 = pd.read_csv("/home/dominik/PycharmProjects/TensorCFR/src/nn/data/history_identifier.csv",index_col=0)
def get_files_in_directory_recursively(rootdir):
	import os
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
	return pd.read_csv(path,index_col=0)

def filter_by_public_state(df=None,public_state=None):
	if df is not None:
		df = df.copy()

		if public_state.__len__() == 3:
			return df.loc[(df["r1"]==int(public_state[0])) & (df["r2"]== int(public_state[1])) & (df["r3"]==int(public_state[2]))]
	else:
		return ValueError

def filter_by_card_combination(df=None,cards=None,player=None):

	import pandas as pd

	cards_filtered = pd.DataFrame()

	if player == 1:
		cards_filtered = df.loc[(df["r1c1"]==int(cards[1])) & (df["r2c1"]== int(cards[4])) & (df["r3c1"]==int(cards[7]))]


	if player == 2:
		cards_filtered = df.loc[(df["r1c2"]==int(cards[1])) & (df["r2c2"]== int(cards[4])) & (df["r3c2"]==int(cards[7]))]



	return cards_filtered

def seed_to_sum_cfv_per_infoset(df=None):
	#TODO
	pass

def calc_append_cfv_p2(dir=""):

	##TODO divide by sum of cf reaches of p1 and multiply by sum of cf reach of p2

	import os
	import pandas as pd

	if not os._exists(dir):
		raise ValueError

	else:

		file_list = get_files_in_directory_recursively(dir)
		seed_list = [load_seed_from_filepath(seed) for seed in file_list]
		i = 0
		for seed in seed_list:
			print("seed {}".format(i+1))
			cfv_p2 = pd.Series((seed["\t nodal_cf_value"]/seed["\t nodal_reach_1"]) * (-1) * seed["\t nodal_reach_2"],name="\t nodal_cf_value_p2")
			seed = pd.concat([seed,cfv_p2],axis=1)
			seed.to_csv(path_or_buf=file_list[i])
			i+=1

def seed_to_ranges_per_public_state(df=None):
	##TODO
	mask = load_input_mask()
	out = load_output_mask()
	hist_id = load_history_identifier()
	public_states_list = [(x,y,z) for x in [0, 1, -1] for y in [0, 1, -1] for z in [0, 1, -1]]

	for public_state in public_states_list:

		df_by_public_state = filter_by_public_state(hist_id,public_state)
		#print(public_state)
		#print(df_by_public_state.shape)

		#for player in [1,2]:

		for cards in mask.columns[3:123]:
			## for player 1

			cards_df = filter_by_card_combination(df_by_public_state,cards,1)
			#print("infoset {} contains {} histories".format(cards,cards_df.shape[0]))


			if cards_df.shape[0] == 1:

				# puts range of p1 in of infoset "cards" of public state "public_state" into mask

				mask.loc["".join(tuple(map(str,public_state))),cards] = float(df.iloc[df.index == cards_df.index[0],9])

				# puts cf of p1 in of infoset "cards" of public state "public_state" into out

				out.loc["".join(tuple(map(str, public_state))), cards] = float(df.iloc[df.index == cards_df.index[0], 7])

			elif cards_df.shape[0] > 1:

				mask.loc["".join(tuple(map(str,public_state))), cards] = float(df.iloc[df.index == cards_df.index[0], 9])

				out.loc["".join(tuple(map(str, public_state))), cards] = sum([float(df.iloc[df.index == cards_df.index[i], 7]) for i in range(0,cards_df.index.__len__())])

			elif cards_df.shape[0] == 0:

				mask.loc["".join(tuple(map(str,public_state))), cards] = 0

				out.loc["".join(tuple(map(str, public_state))), cards] = 0

		for cards in mask.columns[123:]:

			cards_df = filter_by_card_combination(df_by_public_state, cards, 2)

			if cards_df.shape[0] == 1:

				mask.loc["".join(tuple(map(str,public_state))), cards] = float(df.iloc[df.index == cards_df.index[0], 10])

				out.loc["".join(tuple(map(str, public_state))), cards] = float(df.iloc[df.index == cards_df.index[0], 12])


			elif cards_df.shape[0] > 1:

				mask.loc["".join(tuple(map(str,public_state))), cards] = float(df.iloc[df.index == cards_df.index[0], 10])

				out.loc["".join(tuple(map(str, public_state))), cards] = sum([float(df.iloc[df.index == cards_df.index[i], 12]) for i in range(0,cards_df.index.__len__())])

			elif cards_df.shape[0] == 0:

				mask.loc["".join(tuple(map(str,public_state))), cards] = 0

				out.loc["".join(tuple(map(str, public_state))), cards] = 0


	#out_minus = -out

	#out_minus.columns = [column_name+".1" for column_name in out_minus.columns]

	#full_out = pd.concat([out,out_minus],axis=1)

	return mask,out

def build_training_data(data_dir="",num=None):
	#import numpy as np
	import pandas as pd
	file_list = get_files_in_directory_recursively(data_dir)
	seed_list = [load_seed_from_filepath(seed) for seed in file_list]

	## x
	x = pd.DataFrame()
	y= pd.DataFrame()
	i = 1
	max = seed_list.__len__()

	if num is not None and num is int and num < seed_list.__len__():
		max = num


	for seed in seed_list[:max]:
		print("seed:{} of {}".format(i,file_list.__len__()))

		if x.shape == (0,0):

			x,y = seed_to_ranges_per_public_state(seed)

		else:
			new_x,new_y = seed_to_ranges_per_public_state(seed)
			x = pd.concat([x,new_x],axis=0)
			y = pd.concat([y,new_y],axis=0)

		i+= 1

	return x,y


def get_indices_hist_in_infoset(public_state=None,cards=None,player=1):

	hist_id = load_history_identifier()

	mypubstate = filter_by_public_state(df=hist_id,public_state=public_state)

	myinfoset = filter_by_card_combination(df=mypubstate,cards=cards,player=player)

	return myinfoset.index


def extract_first_hist_of_infoset_indices_from_seed():
	import numpy as np
	##TODO
	#mask = load_input_mask()
	out = load_output_mask()
	hist_id = load_history_identifier()
	public_states_list = [(x,y,z) for x in [0, 1, -1] for y in [0, 1, -1] for z in [0, 1, -1]]

	mydict = {}

	for public_state in public_states_list:

		df_by_public_state = filter_by_public_state(hist_id,public_state)

		for cards in out.columns[:120]:
			## for player 1

			cards_df = filter_by_card_combination(df_by_public_state,cards,1)

			out.loc["".join(tuple(map(str, public_state))), cards] = int(cards_df.index[0]) if cards_df.shape[0] > 0 else int(-1)
			#mydict["".join(tuple(map(str,public_state)))+cards] = list(cards_df.index)

		for cards in out.columns[120:]:

			cards_df = filter_by_card_combination(df_by_public_state, cards, 2)
			out.loc["".join(tuple(map(str, public_state))), cards] = int(cards_df.index[0]) if cards_df.shape[0] > 0 else int(-1)
			#mydict["".join(tuple(map(str,public_state)))+cards] = list(cards_df.index)


	return out.astype(int)