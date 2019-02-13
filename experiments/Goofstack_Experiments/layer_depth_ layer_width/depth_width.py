import numpy as np
import pandas as pd
import tensorflow as tf
from src.nn.data import preprocessing_ranges
from keras.models import Model
from keras.layers import Input,Dense,BatchNormalization
from keras.initializers import he_normal,lecun_normal
from keras.optimizers import Adam
import os

from src.utils.other_utils import activate_script


def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = tf.keras.backend.abs(error) < clip_delta

  squared_loss = 0.5 * tf.keras.backend.square(error)
  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

  return tf.where(cond, squared_loss, linear_loss)


def create_model(width=None):

	shape = 243
	input = Input(shape=(shape,))
	#b1 = BatchNormalization()(input)
	dense1 = Dense(units=width,activation="relu",kernel_initializer=he_normal())(input)
	#b2 = BatchNormalization()(dense1)
	dense2 = Dense(units=width,activation="relu",kernel_initializer=he_normal())(dense1)
	#b3 = BatchNormalization()(dense2)
	out = Dense(units=120,activation="linear")(dense2)

	model = Model(inputs=input,outputs=out)

	model.compile(optimizer=Adam(),loss=huber_loss)

	return model


def create_model_n_layers(n_layers=2):

	shape = 243

	if n_layers == 2:
		input = Input(shape=(shape,))
		#b1 = BatchNormalization()(input)
		dense1 = Dense(units=width,activation="relu",kernel_initializer=he_normal())(input)
		#b2 = BatchNormalization()(dense1)
		dense2 = Dense(units=width,activation="relu",kernel_initializer=he_normal())(dense1)
		#b3 = BatchNormalization()(dense2)
		out = Dense(units=120,activation="linear")(dense2)

		model = Model(inputs=input,outputs=out)

		model.compile(optimizer=Adam(),loss=huber_loss)

		return model

	elif n_layers == 3:
		input = Input(shape=(shape,))
		# b1 = BatchNormalization()(input)
		dense1 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(input)
		# b2 = BatchNormalization()(dense1)
		dense2 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense1)

		dense3 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense2)
		# b3 = BatchNormalization()(dense2)
		out = Dense(units=120, activation="linear")(dense3)

		model = Model(inputs=input, outputs=out)

		model.compile(optimizer=Adam(), loss=huber_loss)

		return model

	elif n_layers == 4:
		input = Input(shape=(shape,))
		# b1 = BatchNormalization()(input)
		dense1 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(input)
		# b2 = BatchNormalization()(dense1)
		dense2 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense1)

		dense3 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense2)

		dense4 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense3)
		# b3 = BatchNormalization()(dense2)
		out = Dense(units=120, activation="linear")(dense4)

		model = Model(inputs=input, outputs=out)

		model.compile(optimizer=Adam(), loss=huber_loss)

		return model

	elif n_layers == 5:
		input = Input(shape=(shape,))
		# b1 = BatchNormalization()(input)
		dense1 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(input)
		# b2 = BatchNormalization()(dense1)
		dense2 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense1)

		dense3 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense2)

		dense4 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense3)

		dense5 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense4)
		# b3 = BatchNormalization()(dense2)
		out = Dense(units=120, activation="linear")(dense5)

		model = Model(inputs=input, outputs=out)

		model.compile(optimizer=Adam(), loss=huber_loss)

		return model

	elif n_layers == 6:
		input = Input(shape=(shape,))
		# b1 = BatchNormalization()(input)
		dense1 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(input)
		# b2 = BatchNormalization()(dense1)
		dense2 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense1)

		dense3 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense2)

		dense4 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense3)

		dense5 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense4)

		dense6 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense5)
		# b3 = BatchNormalization()(dense2)
		out = Dense(units=120, activation="linear")(dense6)

		model = Model(inputs=input, outputs=out)

		model.compile(optimizer=Adam(), loss=huber_loss)

		return model

	elif n_layers == 7:
		input = Input(shape=(shape,))
		# b1 = BatchNormalization()(input)
		dense1 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(input)
		# b2 = BatchNormalization()(dense1)
		dense2 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense1)

		dense3 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense2)

		dense4 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense3)

		dense5 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense4)

		dense6 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense5)

		dense7 = Dense(units=width, activation="relu", kernel_initializer=he_normal())(dense6)
		# b3 = BatchNormalization()(dense2)
		out = Dense(units=120, activation="linear")(dense7)

		model = Model(inputs=input, outputs=out)

		model.compile(optimizer=Adam(), loss=huber_loss)

		return model



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
#model.fit(x=x,y=y,batch_size=seed_shape,epochs=200)
if __name__ == "__main__" and activate_script():

	x = pd.read_csv(filepath_or_buffer="~/Desktop/nn_train/input.csv",index_col=0)
	y = pd.read_csv(filepath_or_buffer="~/Desktop/nn_train/target.csv",index_col=0)
	y = y.iloc[:,:120]

	shape = 243

	seed_shape = 27


	width_model_list = []

	for width in [shape*mult for mult in range(1,6)]:
		model = create_model(width)

		model.fit(x=x, y=y, batch_size=seed_shape, epochs=200,validation_split=0.1)

		width_model_list.append(model)

	depth_model_list = []

	for depth in [depth for depth in range(2, 7)]:
		model = create_model_n_layers(depth)

		model.fit(x=x, y=y, batch_size=seed_shape, epochs=200, validation_split=0.1)

		depth_model_list.append(model)

	widthconfigs = [shape*mult for mult in range(1,6)]

	depthconfigs = [depth for depth in range(2, 7)]

	import pickle

	for i in range(widthconfigs.__len__()):
		with open(os.getcwd()+"/"+str(widthconfigs[i])+".pkl"),"wb") as f:
			pickle.dump(width_model_list[i].history.history,f,protocol=pickle.HIGHEST_PROTOCOL)

	for i in range(depthconfigs.__len__()):
		with open(os.getcwd() + "/" + str(depthconfigs[i]) + ".pkl"), "wb") as f:
			pickle.dump(depth_model_list[i].history.history,f,protocol=pickle.HIGHEST_PROTOCOL)

##

