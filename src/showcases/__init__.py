import sys
import tensorflow as tf


sys.path.append('..')
sys.path.append('../utils')

from utils.tensor_utils import print_tensor

if __name__ == '__main__':
	state2IS = tf.Variable([0, 1, 2, 2, 4], name="state2IS")
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print_tensor(sess, state2IS)
