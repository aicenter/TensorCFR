from src.nn.Runner_CNN_IIGS6Lvl10_NPZ import Runner_CNN_IIGS6Lvl10_NPZ
from src.nn.sanity_cnn import SanityCNN
from src.utils.other_utils import get_current_timestamp


class SanityCNNRunner(Runner_CNN_IIGS6Lvl10_NPZ):
	def construct_network(self):
		self.network = SanityCNN(threads=self.args.threads, fixed_randomness=self.fixed_randomness)
		self.network.construct(self.args)
		return self.network

	def run_neural_net(self, ckpt_every=None, ckpt_dir=None):
		dataset_directory = self.args.dataset_directory
		self.create_logdir()

		devset, testset, trainset = self.init_datasets(dataset_directory)
		self.network = self.construct_network()

		if ckpt_dir is None:
			ckpt_dir = self.args.logdir

		for self.epoch in range(self.args.epochs):
			self.train_one_epoch(trainset)
			self.evaluate_devset(devset)

			# checkpoint every `ckpt_every` epochs
			if ckpt_every and ckpt_dir is not None and int(self.epoch) % int(ckpt_every) == 0:
				ckpt_basename = "epoch_{}_{}".format(str(self.epoch), str(get_current_timestamp()))
				self.ckpt_basenames.append(ckpt_basename)
				self.network.save_to_ckpt(ckpt_dir, ckpt_basename)

		ckpt_basename = "final_{}.ckpt".format(get_current_timestamp())
		self.ckpt_basenames.append(ckpt_basename)
		self.network.save_to_ckpt(ckpt_dir, ckpt_basename)  # final model

		self.evaluate_testset(testset)
		self.showcase_predictions(trainset)


if __name__ == '__main__':
	runner = SanityCNNRunner()
	runner.run_neural_net()
