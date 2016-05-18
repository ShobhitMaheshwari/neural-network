import argparse, mnist_loader, random
import numpy as np
import scipy.misc
import matplotlib.animation as animation

def saveImage(name, image):
    """ Plot a single MNIST image."""
    # scipy.misc.toimage(image, cmin=0.0, cmax=255).save('my.png')
    scipy.misc.imsave(name, image)

def get_images(training_set):
    """ Return a list containing the images from the MNIST data
    set. Each image is represented as a 2-d numpy array."""
    flattened_images = training_set[0]
    return zip([np.reshape(f, (-1, 28)) for f in flattened_images], training_set[1])

def main():
    training_set, validation_set, test_set = mnist_loader.load_data()
    images = get_images(training_set)
    images = [(image, digit) for (image, digit) in images if digit == 7]
    idxs = random.sample(xrange(1, len(images)), 2)

    saveImage('x.png', images[idxs[0]][0])
    saveImage('y.png', images[idxs[1]][0])


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                     help='an integer for the accumulator')
    #
    #
    # parser.add_argument('--train', dest="is_training_mode", action="store_true",
    #                     help='Is it a training mode or testing mode?')
    # parser.add_argument('-c', dest="classifier_fname", default="classifier.pickle",
    #                     help='File name of the classifier pickel.')
    #
    # parser.add_argument('--foo', help='foo help', dest="foo")
    # args = parser.parse_args()
    # print(args.integers)
    # print(args.classifier_fname)
    # print(args.is_training_mode)
    # print(args.foo)

    main()

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

    pass