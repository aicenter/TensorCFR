from abc import abstractmethod


class AbstractNN:
	@abstractmethod
	def predict(self, input_tensor):
		return input_tensor * 10