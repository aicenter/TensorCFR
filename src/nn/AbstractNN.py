from abc import abstractmethod


class AbstractNN:
	@abstractmethod
	def predict(self, input_features):
		raise NotImplementedError
