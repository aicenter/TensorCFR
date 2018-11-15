import numpy as np
import tensorflow as tf


if __name__ == '__main__':
	x1 = tf.placeholder(tf.float32)
	c1 = tf.constant(2.0)
	mul_op = tf.multiply(x1, c1)

	x2 = tf.placeholder(tf.float32)
	c2 = tf.constant(2.0)
	div_op = tf.divide(x2, c2)

	y = np.array([1, 2, 3])

	for _ in range(10000):
		with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
			y = session.run(mul_op, feed_dict={x1: y})
			print("Mul: " + str(y))

		with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
			y = session.run(div_op, feed_dict={x2: y})
			print("Div: " + str(y))
