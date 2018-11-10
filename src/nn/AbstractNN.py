from abc import abstractmethod

import tensorflow as tf


class AbstractNN:
	@abstractmethod
	def predict(self, input_features, pass_input_using=None):
		raise NotImplementedError

	@abstractmethod
	def call_saver(self,path):

		if self.saver not in tf.global_variables():

			AbstractNN.construct_saver(self)

		self.saver.save(self.session,path)

	@abstractmethod
	def save_to_ckpt(self, path):
		if str(path).endswith(".ckpt"):
			self.saver.save(self.session, path)
		else:
			self.saver.save(self.session, path + ".ckpt")
		print("Saving to " + path + " successful")

	@abstractmethod
	def restore_from_ckpt(self, path):
		if str(path).endswith(".ckpt"):
			self.saver.restore(self.session, path)
		else:
			self.saver.restore(self.session, path + ".ckpt")
		print("Restoring from "+path +" successful")