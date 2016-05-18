import argparse, mnist_loader, random
import numpy as np
import scipy.misc
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def rand(x, y):
    return [[random.random() for i in range(y)] for j in range(x)]


def ani_frame():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(rand(300,300),cmap='gray',interpolation='nearest')
    im.set_clim([0,1])
    fig.set_size_inches([5,5])
    plt.tight_layout()

    def update_img(n):
        tmp = rand(300,300)
        im.set_data(tmp)
        return im

    #legend(loc=0)
    ani = animation.FuncAnimation(fig,update_img,300,interval=30)
    writer = animation.writers['ffmpeg'](fps=30)

    ani.save('demo.mp4',writer=writer,dpi=100)
    return ani


def saveImage(name, image):
    """ Plot a single MNIST image."""
    scipy.misc.toimage(255*image, cmin=0.0, cmax=255).save(name)
    # scipy.misc.imsave(name, image)


def get_images(training_set):
    """ Return a list containing the images from the MNIST data
    set. Each image is represented as a 2-d numpy array."""
    flattened_images = training_set[0]
    return zip([np.reshape(f, (-1, 28)) for f in flattened_images], training_set[1])


def main():
    training_set, validation_set, test_set = mnist_loader.load_data()
    images = get_images(training_set)
    images = [image for (image, digit) in images if digit == 7]
    idxs = random.sample(xrange(1, len(images)), 2)

    saveImage('x.png', images[idxs[0]])
    saveImage('y.png', images[idxs[1]])
    ani_frame()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                     help='an integer for the accumulator')
    # parser.add_argument('--train', dest="is_training_mode", action="store_true",
    #                     help='Is it a training mode or testing mode?')
    # parser.add_argument('-c', dest="classifier_fname", default="classifier.pickle",
    #                     help='File name of the classifier pickel.')
    # parser.add_argument('--foo', help='foo help', dest="foo")
    # args = parser.parse_args()
    # print(args.integers)
    # print(args.classifier_fname)
    # print(args.is_training_mode)
    # print(args.foo)

    main()