from abc import abstractmethod


class AbstractNN:
	@abstractmethod
	def predict(self, input_features, pass_input_using=None):
		raise NotImplementedError

	def save_to_ckpt(self, ckpt_dir, ckpt_basename):     # Note: `ckpt_dir` shouldn't end with `/`
		if not str(ckpt_basename).endswith(".ckpt"):
			ckpt_basename += ".ckpt"
		filename = "{}/{}".format(ckpt_dir, ckpt_basename)
		self.saver.save(self.session, filename)
		print("Saving to {} successful".format(filename))

	def restore_from_ckpt(self, ckpt_dir, ckpt_basename):     # Note: `ckpt_dir` shouldn't end with `/`
		if not str(ckpt_basename).endswith(".ckpt"):
			ckpt_basename += ".ckpt"
		filename = "{}/{}".format(ckpt_dir, ckpt_basename)
		self.saver.restore(self.session, filename)
		print("Restoring from {} successful".format(filename))
