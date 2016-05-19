import argparse, mnist_loader, random
import numpy as np
import scipy.misc
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def ani_frame(name, images):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    im = ax.imshow(images[0],cmap='gray',interpolation='nearest')
    im.set_clim([0,1])
    fig.set_size_inches([5,5])
    plt.tight_layout()

    def update_img(n):
        im.set_data(images[update_img.index])
        update_img.index += 1
        return im

    update_img.index = 0
    #legend(loc=0)
    ani = animation.FuncAnimation(fig,update_img,len(images) - 1)
    writer = animation.writers['ffmpeg'](fps=2)
    ani.save(name, writer=writer)
    return ani


def saveImage(name, image):
    """ Plot a single MNIST image."""
    scipy.misc.toimage(255*image, cmin=0.0, cmax=255).save('pics/'+name)
    # scipy.misc.imsave(name, image)


def get_images(training_set):
    """ Return a list containing the images from the MNIST data
    set. Each image is represented as a 2-d numpy array."""
    flattened_images = training_set[0]
    return zip([np.reshape(f, (-1, 28)) for f in flattened_images], training_set[1])


def get_centroid(image):
    centroid_x = 0
    centroid_y = 0
    count = 0
    for x in range(0, len(image)):
        for y in range(0, len(image[x])):
            if(image[x][y]):
                centroid_x += x
                centroid_y += y
                count += 1
    if not count:
        return None
    return centroid_x/float(count), centroid_y/float(count)


def get_entropy(image):
    centroid = get_centroid(image)
    if centroid is None:
        return 0
    x_entropy = 0
    y_entropy = 0
    for x in range(0, len(image)):
        for y in range(0, len(image[x])):
            if (image[x][y]):
                x_entropy += abs(x - centroid[0])
                y_entropy += abs(y - centroid[1])
    return x_entropy + y_entropy


def get_max_entropy_images(images):
    max = -1
    max_x = 0
    max_y = 0
    for x in range(0, len(images)):
        for y in range(x+1, len(images)):
            ent = get_entropy(images[x] - images[y])
            if(ent > max):
                print(ent, x, y)
                max = ent
                max_x = x
                max_y = y
    return max_x, max_y


def test():
    a = [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    b = [
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 1]
    ]
    images = [np.array(a),np.array(b)]
    max = -1
    max_x = 0
    max_y = 0
    for x in range(0, len(images)):
        for y in range(x + 1, len(images)):
            ent = get_entropy(images[x] - images[y])
            if (ent > max):
                print(ent, x, y)
                max = ent
                max_x = x
                max_y = y
    return x, y


def main():
    training_set, validation_set, test_set = mnist_loader.load_data()
    images = get_images(training_set)
    images = [image for (image, digit) in images if digit == 5]
    # images = [image for (image, digit) in images]
    # idxs = random.sample(xrange(1, len(images)), 2)
    x,y = get_max_entropy_images(images[0:200])
    saveImage("x.png", images[x])
    saveImage("y.png", images[y])
    # ani_frame('demo.mp4', images[0:100])


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
    # test()