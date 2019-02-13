import numpy as np
import pandas as pd
import tensorflow as tf
from src.nn.data import preprocessing_ranges
from keras.models import Model
from keras.layers import Input,Dense,BatchNormalization
from keras.initializers import he_normal,lecun_normal
from keras.optimizers import Adam
import os
from sklearn.preprocessing import normalize

def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = tf.keras.backend.abs(error) < clip_delta

  squared_loss = 0.5 * tf.keras.backend.square(error)
  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

  return tf.where(cond, squared_loss, linear_loss)

##TODO write indexing method that maps predictions for each public state back to what TensorCFR expects


## 1. you need to create the counterfactual values of player 2 with
## src.nn.data.preprocessing_ranges.calc_append_cfv_p2("directory where your seeds are")
# # then you create the trainign data with
# x,y = preprocessing_ranges.build_training_data('your directory of seeds')

## TODO figure out how to save the keras model to a tensorflow graph



x,y = preprocessing_ranges.build_training_data(os.getcwd()+"/src/nn/data/out/300119_seeds")


#test = pd.read_csv(filepath_or_buffer=os.getcwd()+"/src/nn/data/out/IIGS6_gambit_flattened/4_datasets/IIGS6_s1_bf_ft_gambit_flattened-2019-01-30_132724-ad=250,ts=1000,td=10/nodal_dataset_seed_0.csv",index_col=0)
#x = pd.read_csv(filepath_or_buffer=os.getcwd()+"/src/nn/data/out/nn data/x.csv",index_col=0)
#y = pd.read_csv(filepath_or_buffer=os.getcwd()+"/src/nn/data/out/nn data/y.csv",index_col=0)
x = pd.read_csv(filepath_or_buffer="~/Desktop/nn_train/input.csv",index_col=0)
y = pd.read_csv(filepath_or_buffer="~/Desktop/nn_train/target.csv",index_col=0)

shape = 243

seed_shape = 27

input = Input(shape=(shape,))
#b1 = BatchNormalization()(input)
dense1 = Dense(1024,activation="selu",kernel_initializer=lecun_normal())(input)
#b2 = BatchNormalization()(dense1)
#dense2 = Dense(1024,activation="selu",kernel_initializer=lecun_normal())(b2)
#b3 = BatchNormalization()(dense2)
out = Dense(120,activation="linear")(dense1)

model = Model(inputs=input,outputs=out)

model.compile(optimizer=Adam(),loss="mse")

model.fit(x=x,y=y.iloc[:,:120],batch_size=27,epochs=100)

##
import pandas as pd
dat = pd.read_csv("/home/dominik/PycharmProjects/TensorCFR/src/nn/data/out/IIGS6_gambit_flattened/4_datasets/IIGS6_s1_bf_ft_gambit_flattened-2019-01-25_161617-ad=250,ts=1000,td=10/nodal_dataset_seed_0.csv",index_col=0)
from src.nn.data.preprocessing_ranges import *
mask,out,hist_id = load_input_mask(),load_output_mask(),load_history_identifier()

import tensorflow as tf

graph = tf.Graph()

with graph.as_default():


  a = tf.constant(2,dtype=tf.int32)

  b = tf.constant(5,dtype=tf.int32)

  def c():
    return a*b


with tf.Session() as sess:

  a = tf.constant(2, dtype=tf.int32)

  b = tf.constant(5, dtype=tf.int32)


  def c():
    return a * b


  print(sess.run(c))


# Build a graph.
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

# Launch the graph in a session.
sess = tf.Session()

# Evaluate the tensor `c`.
print(sess.run(c))


