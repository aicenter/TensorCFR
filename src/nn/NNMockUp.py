from src.nn.AbstractNN import AbstractNN


class NNMockUp(AbstractNN):
	def predict(self, input_features):
		return input_features * 10