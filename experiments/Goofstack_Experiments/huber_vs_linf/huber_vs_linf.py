
import pandas as pd
import os
from src.nn.data.postprocessing_ranges import huber_loss,load_nn,linf_loss
#from experiments.Goofstack_Experiments.layer_depth_layer_width import create_model()
from keras.layers import Dense,Input
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import he_normal

from src.utils.other_utils import activate_script

shape = 243

seed_shape = 27

def create_loss(loss=huber_loss):

	shape = 243
	input = Input(shape=(shape,))
	#b1 = BatchNormalization()(input)
	dense1 = Dense(units=shape*5,activation="relu",kernel_initializer=he_normal())(input)
	#b2 = BatchNormalization()(dense1)
	dense2 = Dense(units=shape*5,activation="relu",kernel_initializer=he_normal())(dense1)

	dense3 = Dense(units=shape * 5, activation="relu", kernel_initializer=he_normal())(dense2)

	dense4 = Dense(units=shape * 5, activation="relu", kernel_initializer=he_normal())(dense3)
	#b3 = BatchNormalization()(dense2)
	out = Dense(units=120,activation="linear")(dense4)

	model = Model(inputs=input,outputs=out)

	model.compile(optimizer=Adam(),loss=loss)

	return model

def create_linf(width=None):

	shape = 243
	input = Input(shape=(shape,))
	#b1 = BatchNormalization()(input)
	dense1 = Dense(units=width*5,activation="relu",kernel_initializer=he_normal())(input)
	#b2 = BatchNormalization()(dense1)
	dense2 = Dense(units=width*5,activation="relu",kernel_initializer=he_normal())(dense1)

	dense3 = Dense(units=width * 5, activation="relu", kernel_initializer=he_normal())(dense2)

	dense4 = Dense(units=width * 5, activation="relu", kernel_initializer=he_normal())(dense3)
	#b3 = BatchNormalization()(dense2)
	out = Dense(units=120,activation="linear")(dense4)

	model = Model(inputs=input,outputs=out)

	model.compile(optimizer=Adam(),loss=linf_loss)

	return model

huber_linf_list = []

if __name__== "__main__" and activate_script():

	x = pd.read_csv(filepath_or_buffer="~/Desktop/nn_train/input.csv",index_col=0)
	y = pd.read_csv(filepath_or_buffer="~/Desktop/nn_train/target.csv",index_col=0)
	y = y.iloc[:,:120]

	huber_linf_list = []

	losslist = [huber_loss,linf_loss]

	huber = create_loss(huber_loss)

	huber.fit(x=x, y=y, batch_size=seed_shape, epochs=100,validation_split=0.1,metrics=[linf_loss])

	huber_linf_list.append(huber)

	linf = create_loss(linf_loss)

	linf.fit(x=x, y=y, batch_size=seed_shape, epochs=100, validation_split=0.1, metrics=[huber_loss])

	huber_linf_list.append(linf)


##

def plot_network_losses(model_list,scale="log"):
	from matplotlib import pyplot as plt
	from numpy import arange

	x = arange(200)

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