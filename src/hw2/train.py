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


def main():
	training_set, validation_set, test_set = mnist_loader.load_data_wrapper()

	training_set = get_modified_training_set(training_set)
	net = src.network.Network([784, 30, 10])
	net.SGD(training_set, 30, 10, 3.0, test_data=test_set)

	pass

if __name__=="__main__":
	# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	# digit = np.zeros((10,1))
	# digit[0]=[1]
	# digits = [(x, y) for (x, y) in training_data if np.array_equal(y, digit)]
	# idxs = random.sample(xrange(1, len(digits)), 2)
	# data = [digits[idx] for idx in idxs]
	# data = [x for (x, y) in data]
	# img = Image.fromarray(255*np.reshape(data[0], (28, 28)), 'L')
	# img.save('my.png')

	# training_data, validation_data, test_data = mnist_loader.load_data()
	# digit = np.zeros((10,1))
	# print((training_data[0].shape))
	# digits = [(x, y) for (x, y) in zip(training_data[0], training_data[1]) if y==1]
	# idxs = random.sample(xrange(1, len(digits)), 2)
	# data = [digits[idx] for idx in idxs]
	# data = [x for (x, y) in data]
	# img = Image.fromarray(255*np.reshape(data[0], (28, 28)), 'L')
	# img.save('my.png')
	main()