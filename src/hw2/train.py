from graph import Graph
import sys
sys.path.append('../..')
import src.mnist_loader as mnist_loader
import numpy as np
import random
import src.network

def get_modified_training_set(training_set, samples_per_digit = 100):
	modified_training_set = []
	for digit in range(0, 10):
		out = np.zeros((10, 1))
		out[digit] = [1]
		digits = [(x, y) for (x, y) in training_set if np.array_equal(y, out)]
		tmp = Graph(digits[0: samples_per_digit]).get_modified_input()
		# tmp = digits[0: samples_per_digit]
		modified_training_set.extend(tmp)
		print("modifying dataset {digit}. Now adding {count} more examples".format(digit=digit, count=len(tmp)))
	random.shuffle(modified_training_set)
	return modified_training_set


def normalize(n, lst):
	# samples = []
	# for x in lst:
	# 	samples.extend([x[0].get_training_example()])
	# return samples

	count = 0
	for x in lst:
		count += x[1]
	tmp_lst = []
	for x in lst:
		tmp_lst.append((x[0], int(x[1] * n / float(count))))

	samples = []
	for x in tmp_lst:
		samples.extend([x[0].get_training_example()] * x[1])
	return samples

def take_best_n_samples(n, lst):
	samples = []
	count = 0
	for x in lst:
		if x[1] + count > n:
			samples.extend([x[0].get_training_example()] * (n - count))
			break
		else:
			samples.extend([x[0].get_training_example()] * x[1])
			count += x[1]

	# samples = []
	# for x in lst:
	# 	samples.extend([x[0].get_training_example()])
	return samples

def get_modified_training_set2(training_set, samples_per_digit = 100):
	modified_training_set = []
	dct = {}
	max = float('-inf')
	for digit in range(0, 10):
		out = np.zeros((10, 1))
		out[digit] = [1]
		digits = [(x, y) for (x, y) in training_set if np.array_equal(y, out)]
		count, lst = Graph(digits[0: samples_per_digit]).get_modified_input2()
		dct[digit] = lst
		print("modifying dataset {digit}. Now adding {count} more examples".format(digit=digit, count=count))
		if count > max:
			max = count

	for x in dct:
		best_n_samples = normalize(65931, dct[x])#hardcoded value
		modified_training_set.extend(best_n_samples)

	random.shuffle(modified_training_set)
	return modified_training_set


def main():
	training_set, validation_set, test_set = mnist_loader.load_data_wrapper()

	training_set = get_modified_training_set2(training_set)
	print (len(training_set))
	net = src.network.Network([784, 30, 10])
	net.SGD(training_set, 10, 10, 3.0, test_data=test_set)

	pass

if __name__=="__main__":
	main()