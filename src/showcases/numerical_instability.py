import numpy as np
import tensorflow as tf


if __name__ == '__main__':
	# x1 = tf.placeholder(tf.float32)
	# c1 = tf.placeholder(tf.float32)
	# mul_op = tf.multiply(x1, c1)
	#
	# x2 = tf.placeholder(tf.float32)
	# c2 = tf.placeholder(tf.float32)
	# div_op = tf.divide(x2, c2)

	y = np.random.rand(3)

	add_sub_vector = np.random.rand(3)
	print("Before")
	print(y)

	for prime in [7, 9, 11, 12, 13]:
		for _ in range(1000):
			graph1 = tf.Graph()
			with graph1.as_default():
				x1 = tf.placeholder(tf.float32)
				c1 = tf.placeholder(tf.float32)
				mul_op = tf.multiply(x1, c1)

				with tf.Session() as session:
					y = session.run(mul_op, feed_dict={x1: y, c1: prime})
					# print("Mul: " + str(y))
				    #y += add_sub_vector
			del graph1

			graph2 = tf.Graph()
			with graph2.as_default():
				x2 = tf.placeholder(tf.float32)
				c2 = tf.placeholder(tf.float32)
				div_op = tf.divide(x2, c2)

				with tf.Session() as session:
					#y -= add_sub_vector
					y = session.run(div_op, feed_dict={x2: y, c2: prime})
					# print("Div: " + str(y))
			del graph2
	print("After")
	print(y)
