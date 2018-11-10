from abc import abstractmethod


class AbstractNN:
	@abstractmethod
	def predict(self, input_features, pass_input_using=None):
		raise NotImplementedError

	def save_to_ckpt(self, path):
		if str(path).endswith(".ckpt"):
			self.saver.save(self.session, path)
		else:
			self.saver.save(self.session, path + ".ckpt")
		print("Saving to " + path + " successful")

	def restore_from_ckpt(self, path):
		if str(path).endswith(".ckpt"):
			self.saver.restore(self.session, path)
		else:
			self.saver.restore(self.session, path + ".ckpt")
		print("Restoring from " + path + " successful")
