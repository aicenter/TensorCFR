#!/usr/bin/env python3

import tensorflow as tf


def scoping(fn, scope1, scope2, vals):
	with fn(scope1):
		a = tf.Variable(vals[0], name='a')
		b = tf.get_variable('b', initializer=vals[1])
		c = tf.constant(vals[2], name='c')
		with fn(scope2):
			d = tf.add(a * b, c, name='res')

		print("{}\n".format('\n  '.join([scope1, a.name, b.name, c.name, d.name])))
	return d


if __name__ == '__main__':
	d1 = scoping(tf.variable_scope, 'scope_vars', 'res', [1, 2, 3])
	d2 = scoping(tf.name_scope,     'scope_name', 'res', [1, 2, 3])
	with tf.Session() as sess:
		writer = tf.summary.FileWriter('logs', sess.graph)
		sess.run(tf.global_variables_initializer())
		print(sess.run([d1, d2]))
		writer.close()
