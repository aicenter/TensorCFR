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

from src.nn.data.postprocessing_ranges import load_nn
from src.nn.data.preprocessing_ranges import *
from src.utils.other_utils import activate_script


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


if __name__ == "__main__" and activate_script():
  list= preprocessing_ranges.get_files_in_directory_recursively(os.getcwd()+"/src/nn/data/out/300119_seeds")

  mytargets = pd.read_csv("/home/dominik/Desktop/nn_train/target_240.csv",index_col=0)
  myinput = pd.read_csv("/home/dominik/Desktop/nn_train/input_240.csv",index_col=0)

  nn = load_nn("/home/dominik/PycharmProjects/TensorCFR/experiments/Goofstack_Experiments/exploitability_vs_network_error_per_epoch/nn_300/240out.hdf5")
  ##
  for i in range(myinput.shape[0]):
    pred = nn.predict(myinput.iloc[i,:].values.reshape(1,243),batch_size=1)

    print(sum(np.squeeze(pred*myinput.iloc[i,3:].values.reshape(1,240))))

  ##
  x1 = pd.read_csv(filepath_or_buffer=list[0],index_col=0)
  mypublicstate = filter_by_public_state(load_history_identifier(),public_state=(1,1,1))
  myinfset = filter_by_card_combination(mypublicstate,"(5, 4, 3)",1)


  myx = np.concatenate((x1.loc[:,'\t reach_1'].values.reshape(14400,1),x1.loc[:, '\t reach_2'].values.reshape(14400,1)),axis=0)
  myy = x1.loc[:, '\t nodal_exp_value']


  #test = pd.read_csv(filepath_or_buffer=os.getcwd()+"/src/nn/data/out/IIGS6_gambit_flattened/4_datasets/IIGS6_s1_bf_ft_gambit_flattened-2019-01-30_132724-ad=250,ts=1000,td=10/nodal_dataset_seed_0.csv",index_col=0)
  #x = pd.read_csv(filepath_or_buffer=os.getcwd()+"/src/nn/data/out/nn data/x.csv",index_col=0)
  #y = pd.read_csv(filepath_or_buffer=os.getcwd()+"/src/nn/data/out/nn data/y.csv",index_col=0)
  x = pd.read_csv(filepath_or_buffer="~/Desktop/nn_train/input.csv",index_col=0)
  y = pd.read_csv(filepath_or_buffer="~/Desktop/nn_train/target.csv",index_col=0)

  shape = myx.shape[0]

  seed_shape = 1

  input = Input(shape=(shape,))
  #b1 = BatchNormalization()(input)
  dense1 = Dense(shape,activation="relu",kernel_initializer=he_normal())(input)
  #b2 = BatchNormalization()(dense1)
  #dense2 = Dense(1024,activation="selu",kernel_initializer=lecun_normal())(b2)
  #b3 = BatchNormalization()(dense2)
  out = Dense(myy.shape[0],activation="linear")(dense1)

  model = Model(inputs=input,outputs=out)

  model.compile(optimizer=Adam(),loss="mse")

  model.fit(x=myx.reshape(1,28800),y=myy.values.reshape(1,14400),batch_size=seed_shape,epochs=1)

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


