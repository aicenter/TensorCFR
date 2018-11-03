from abc import abstractmethod


class AbstractNN:
	@abstractmethod
	def predict(self, input_features, pass_input_using=None):
		raise NotImplementedError
