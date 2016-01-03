import os
import math
import scipy.io as matlabIo
import skimage.data
import random
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import matplotlib.pyplot as plt

def make_distance_map(image_shape,
                      centres):

    # Distance transform
    f = np.ones(image_shape, bool)
    for idx in xrange(centres.shape[1]):
        x = int(min(max(0, centres[0, idx]), image_shape[0] - 1))
        y = int(min(max(0, centres[1, idx]), image_shape[1] - 1))

        f[x, y] = False

    distance_map = distance_transform_edt(f)
    return distance_map


def plot_samples(samples,
                 labels,
                 img):
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)

    neg = [samples[idx] for idx in range(len(samples)) if labels[idx]==0]
    negplt, = plt.plot([v for u,v in neg],[u for u,v in neg], 'r.', markersize=5, mew=2)

    pos = [samples[idx] for idx in range(len(samples)) if labels[idx]==1]
    posplt, = plt.plot([v for u,v in pos],[u for u,v in pos], 'g.', markersize=5, mew=2)

    plt.legend([negplt, posplt], ['Negative samples', 'Positive samples'])
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    plt.rcParams['figure.figsize'] = 15,15


def generate(data_type="train",
             box_size=29,
             sample_radius=2,
             enable_plotting=False):

    root = os.path.abspath('dataset/bloodcells/')

    if not box_size % 2 == 1:
        raise Exception("Please choose odd box_size")

    if data_type == "train":
        im_path = root + '/bloodcells_1.png'
        centres_path = root + '/annotations/bloodcells_1.mat'
        neg_path = root + '/annotations/negative_samples_1.mat'
    elif data_type == "validation":
        im_path = root + '/bloodcells_2.png'
        centres_path = root + '/annotations/bloodcells_2.mat'
        neg_path = root + '/annotations/negative_samples_2.mat'
    else:
        raise Exception('Data type is either "train" or "validation"')

    image_border_margin = int(math.floor(box_size / 2) + 1)

    # Load an image
    img = skimage.data.load(im_path)
    img = img.mean(axis=2)

    # Load image centres
    matlab_data = matlabIo.loadmat(centres_path)
    centres = matlab_data['centres']
    distance_to_positive = make_distance_map(img.shape, centres)

    # Load negative samples
    matlab_data = matlabIo.loadmat(neg_path)
    negative_centres = matlab_data['centres']
    distance_to_negative = make_distance_map(img.shape, negative_centres)


    # Positive samples within sample_radius from a cell centre.
    samples = []
    labels = []
    for x in xrange(image_border_margin, img.shape[0] - image_border_margin):
        for y in xrange(image_border_margin, img.shape[1] - image_border_margin):
            if (distance_to_positive[x][y] < sample_radius):
                samples.append((x, y))
                labels.append(1)

    # Negative samples within sample_radius from negative annotated points
    nbr_positives = len(samples)

    for x in xrange(image_border_margin, img.shape[0] - image_border_margin):
        for y in xrange(image_border_margin, img.shape[1] - image_border_margin):
            if (distance_to_negative[x][y] < sample_radius):
                samples.append((x, y))
                labels.append(0)


    # Random negative samples at distance > sample_radius from cell centres
    offset = len(samples)
    while len(samples) - offset < nbr_positives:
        x = random.randint(image_border_margin, img.shape[0] - image_border_margin)
        y = random.randint(image_border_margin, img.shape[1] - image_border_margin)

        d = distance_to_positive[x, y]

        if d > sample_radius:
            samples.append((x, y))
            labels.append(0)


    if enable_plotting:
        plot_samples(samples, labels, img)

    return samples, labels, img
