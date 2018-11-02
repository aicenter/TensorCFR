from src.nn.AbstractNN import AbstractNN


class NNMockUp(AbstractNN):
	def predict(self, input_tensor):
		return input_tensor * 10