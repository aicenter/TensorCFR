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

#dat = pd.read_csv("/home/dominik/PycharmProjects/TensorCFR/src/nn/data/out/IIGS6_gambit_flattened/4_datasets/IIGS6_s1_bf_ft_gambit_flattened-2019-01-25_161617-ad=250,ts=1000,td=10/nodal_dataset_seed_0.csv",index_col=False)

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

def load_history_identifier():
	from pandas import read_csv
	import os
	return read_csv(os.getcwd()+"/src/nn/data/history_identifier.csv",index_col=0)

def filter_for_public_state(df=None,public_state=""):
	pass

def sum_cfv_per_infoset(df=None):
	pass

def seed_to_ranges_per_public_state(df=None):
	raw = load_input_mask()
	hist_id = load_history_identifier()

def build_training_data(data_dir=""):
	file_list = get_files_in_directory_recursively(data_dir)
