from src.nn.Runner_CNN_IIGS6Lvl10_NPZ import Runner_CNN_IIGS6Lvl10_NPZ
from src.nn.sanity_cnn import SanityCNN


class SanityCNNRunner(Runner_CNN_IIGS6Lvl10_NPZ):
	def construct_network(self):
		self.network = SanityCNN(threads=self.args.threads, fixed_randomness=self.fixed_randomness)
		self.network.construct(self.args)
		return self.network

	def run_neural_net(self, ckpt_every=None, ckpt_dir=None):
		pass


if __name__ == '__main__':
	runner = SanityCNNRunner()
	runner.run_neural_net()
