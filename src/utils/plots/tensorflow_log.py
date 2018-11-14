import argparse

import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == '__main__':
	# Example usage is in `tensorflow_log.sh`.
	parser = argparse.ArgumentParser(description='Plot graphs from TensorFlow\'s log files. For example usage see \'tensorflow_log.sh\'.')
	parser.add_argument('plot_by_TF_tag', default=None, type=str, help="Tensorflow's tag in TensorBoard. E.g. 'train/loss'.")
	parser.add_argument('output_file', default=None, type=str, help="Output filename of a picture.")
	parser.add_argument('--xlabel', default='XLABEL', type=str, help="Label of X axes.")
	parser.add_argument('--ylabel', default='YLABEL', type=str, help="Label of Y axes.")
	parser.add_argument('--ylim_bottom', default=0.06, type=float)
	parser.add_argument('--ylim_top', default=0.2, type=float)
	parser.add_argument('--title', default='TITLE', type=str, help="Title of a graph.")
	parser.add_argument('--curves_names', default=None, type=str, nargs='+', help="An array of names of curves in a figure")
	parser.add_argument('--tensorflow_log', default=None, type=str, nargs='+', help="An array of TensorFlow's log files.")

	args = parser.parse_args()

	graph_logs_values = list()

	for log_file_path in args.tensorflow_log:
		logs_values = dict()
		for e in tf.train.summary_iterator(log_file_path):
			for v in e.summary.value:
				if v.tag not in logs_values.keys():
					logs_values[v.tag] = {
						'steps': [e.step],
						'values': [v.simple_value]
					}
				else:
					logs_values[v.tag]['steps'].append(e.step)
					logs_values[v.tag]['values'].append(v.simple_value)
		graph_logs_values.append(logs_values)

	for idx, graph_log_values in enumerate(graph_logs_values):
		plt.plot(graph_log_values[args.plot_by_TF_tag]['steps'], graph_log_values[args.plot_by_TF_tag]['values'], linewidth=0.5)

	plt.xlabel(args.xlabel)
	plt.ylabel(args.ylabel)
	if args.ylim_bottom is not None and args.ylim_top is not None:
		plt.ylim(args.ylim_bottom, args.ylim_top) # set the y axis to be shown in the fixed interval
	plt.legend(args.curves_names)
	plt.title(args.title)
	plt.savefig(args.output_file)
