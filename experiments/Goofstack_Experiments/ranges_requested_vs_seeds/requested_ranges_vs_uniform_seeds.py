#!/usr/bin/env python3
import tensorflow as tf

from src.algorithms.tensorcfr_nn.TensorCFR_Goofstack import TensorCFR_Goofstack
from src.algorithms.tensorcfr_best_response.ExploitabilityByTensorCFR import ExploitabilityByTensorCFR
from src.algorithms.tensorcfr_best_response.TensorCFR_BestResponse import TensorCFR_BestResponse
from src.domains.available_domains import get_domain_by_name
from src.nn.data.postprocessing_ranges import load_nn
from src.utils.other_utils import activate_script
from src.utils.tf_utils import print_tensors
import numpy as np
from src.nn.data.preprocessing_ranges import get_files_in_directory_recursively
import os


if __name__ == '__main__' and activate_script():

	domain = get_domain_by_name("IIGS6_gambit_flattened")

	nn = load_nn()

	mysteps = 1000

	print("expl for {}".format(nn_epoch))

	tensorcfr = TensorCFR_Goofstack(domain,load_nn(nn_epoch),trunk_depth=10)

	tensorcfr.run_cfr(total_steps=mysteps, delay=5)

	print("tensorcfr iterations done..")



	##TODO for each iteration save ranges


