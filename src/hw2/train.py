import imagetools
import sys
sys.path.append('../..')
import src.mnist_loader as mnist_loader
import numpy as np

def main():
	training_set, validation_set, test_set = mnist_loader.load_data_wrapper()
	for digit in range(0, 10):
		out = np.zeros((10,1))
		out[digit]=[1]
		digits = [(x, y) for (x, y) in training_set if np.array_equal(y, out)]
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