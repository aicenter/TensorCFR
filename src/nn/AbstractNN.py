from abc import abstractmethod


class AbstractNN:
	@abstractmethod
	def predict(self, input_tensor):
		raise NotImplementedError
