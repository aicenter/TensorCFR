import numpy as np
import tensorflow as tf

# Input Tensor
myInputTensor = tf.ones(dtype=tf.float32, shape=[10])  # In your case, this would be the results of some ops

output = myInputTensor * 5.0

with tf.Session() as sess:
	print(sess.run(output))  # == 5.0, using the Tensor value
	myNumpyData = np.zeros(10)
	print(sess.run(output, {myInputTensor: myNumpyData}))  # == 0.0 * 5.0 = 0.0, using the np value
