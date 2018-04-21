#!/usr/bin/env python3
import datetime
import os
import re

import tensorflow as tf


def set_up_tensorboard(session, hyperparameters, basename="tensorcfr", domain_name="domain01"):
	log_dir = "logs/{}-{}-{}-{}".format(
			basename,
			domain_name,
			datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
			",".join(
					("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
					 for key, value in sorted(hyperparameters.items()))).replace("/", "-")
	)
	if not os.path.exists("logs"):
		os.mkdir("logs")
	with tf.variable_scope("tensorboard_operations"):
		summary_writer = tf.contrib.summary.create_file_writer(log_dir, flush_millis=10 * 1000)
		with summary_writer.as_default():
			tf.contrib.summary.initialize(session=session, graph=session.graph)
