import os

from src.commons.constants import PROJECT_ROOT
from src.nn.Runner_CNN_IIGS6Lvl10_NPZ import Runner_CNN_IIGS6Lvl10_NPZ
from src.nn.data.DatasetFromNPZ import DatasetFromNPZ
from src.nn.sanity_cnn import SanityCNN
from src.utils.other_utils import get_current_timestamp


class SanityCNNRunner(Runner_CNN_IIGS6Lvl10_NPZ):
	def add_arguments_to_argparser(self):
		self.argparser.add_argument("--batch_size", default=2, type=int, help="Batch size.")
		self.argparser.add_argument("--dataset_directory", default="./",
		                            help="Relative path to dataset folder.")
		self.argparser.add_argument("--extractor", default="C-6", type=str,
		                            help="Description of the feature extactor architecture.")
		self.argparser.add_argument("--regressor", default="C-6", type=str,
		                            help="Description of the value regressor architecture.")
		self.argparser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
		self.argparser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
		# self.argparser.add_argument("--ckpt_every", default=2, type=float, help="Checkpoint every `ckpt_every` epochs.")
		# self.argparser.add_argument("--ckpt_dir", default=None, type=str, help="Checkpoint directory with model to restore.")
		# self.argparser.add_argument("--ckpt_basename", default=None, type=str, help="Checkpoint name with model to restore.")
	@staticmethod
	def datasets_from_npz(dataset_directory, script_directory):
		p = os.path.join(PROJECT_ROOT, 'src', 'nn', "sanity_dataset.npz")
		trainset = DatasetFromNPZ(p)
		devset = DatasetFromNPZ(p)
		testset = DatasetFromNPZ(p)
		return devset, testset, trainset

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
