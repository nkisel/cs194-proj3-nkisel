
from matplotlib.pyplot import get_current_fig_manager
import cv2
import scipy as sp
import scipy.signal as sps
import scipy.ndimage.interpolation as spi
import skimage.io as skio
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
from skimage.color import rgb2gray

nick = skio.imread("img/nick_gigachad.png")
chad = skio.imread("img/gigachad.png")

def show(img):
    skio.imshow(img)
    skio.show()

def save(img, name):
    skio.imsave(name, img)

def gather(image, num_points):
    """ Choose NUM_POINTS on IMAGE and save them. """
    show(image)
    points = []
    points.extend([(0, 0), (image.shape[0], 0), (0, image.shape[1]), (image.shape[0], image.shape[1])])
    points = plt.ginput(num_points, 0)

    plt.close()
    pickle_name = re.split("\.", image_name)[0] + ".p"
    pickle.dump(points, open(pickle_name, "wb"))

gather(chad, 3)
