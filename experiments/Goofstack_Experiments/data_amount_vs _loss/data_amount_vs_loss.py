import numpy as np
import pandas as pd
import tensorflow as tf
from src.nn.data import preprocessing_ranges
from keras.models import Model
from keras.layers import Input,Dense,BatchNormalization
from keras.initializers import he_normal,lecun_normal
from keras.optimizers import Adam
import os
from src.nn.data.postprocessing_ranges import huber_loss,load_nn
from experiments.Goofstack_Experiments.layer_depth_vs_width import plot_losses

from src.utils.other_utils import activate_script

shape = 243

seed_shape = 27


if __name__== "__main__" and activate_script():

	x = pd.read_csv(filepath_or_buffer="~/Desktop/nn_train/input.csv",index_col=0)
	y = pd.read_csv(filepath_or_buffer="~/Desktop/nn_train/target.csv",index_col=0)
	y = y.iloc[:,:120]
	model = load_nn("/home/dominik/PycharmProjects/TensorCFR/300.hdf5")

	data_amount_model_list = []

	for amount in [seed_shape*i for i in range(20,400,80)]:

		model = load_nn("/home/dominik/PycharmProjects/TensorCFR/300.hdf5")

		model.fit(x=x.iloc[:amount,:], y=y.iloc[:amount,:], batch_size=seed_shape, epochs=100,validation_split=0.1)

		data_amount_model_list.append(model)





	##

def plot_network_losses(model_list,scale="log"):
	from matplotlib import pyplot as plt
	from numpy import arange

	x = arange(200)
	#plt.plot([arange(200) for i in range(width_model_list.__len__())],[np.array(model_list[i].history.history['loss']).shape for i in range(model_list.__len__)])
	#plt.plot(x,[model_list[i].history.history['loss'] for i in range(model_list.__len__())])



	plt.subplot(2, 1, 1)
	plt.plot(x, model_list[0].history.history['loss'])
	plt.plot(x, model_list[1].history.history['loss'])
	plt.plot(x, model_list[2].history.history['loss'])
	plt.plot(x, model_list[3].history.history['loss'])
	plt.plot(x, model_list[4].history.history['loss'])
	plt.title('Error by width of layers (x times input size)')
	plt.ylabel('train loss')
	plt.legend(["1x","2x","3x","4x","5x"])

	plt.subplot(2, 1, 2)
	plt.plot(x, model_list[0].history.history['val_loss'])
	plt.plot(x, model_list[1].history.history['val_loss'])
	plt.plot(x, model_list[2].history.history['val_loss'])
	plt.plot(x, model_list[3].history.history['val_loss'])
	plt.plot(x, model_list[4].history.history['val_loss'])
	plt.xlabel('epoch')
	plt.xscale(scale)
	plt.ylabel('val_loss')
	plt.legend(["1x", "2x", "3x", "4x", "5x"])

	plt.show()