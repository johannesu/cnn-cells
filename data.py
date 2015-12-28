import os
import math
import scipy.io as matlabIo
import skimage.data
import random
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt


# Can be turned into a generator to speed up things
def generate(data_type="train",
             box_size=29,
             positive_sample_radius=2,
             negative_sample_radius=6):

    root = os.path.abspath('dataset/bloodcells/')

    if not box_size % 2 == 1:
        raise Exception("Please choose odd box_size")

    if data_type == "train":
        im_path = root + '/bloodcells_1.png'
        centres_path = root + '/annotations/bloodcells_1.mat'
    elif data_type == "validation":
        im_path = root + '/bloodcells_2.png'
        centres_path = root + '/annotations/bloodcells_2.mat'
    else:
        raise Exception('Data type is either "train" or "validation"')

    image_border_margin = int(math.floor(box_size / 2) + 1)

    # Load an image
    img = skimage.data.load(im_path)
    img = img.mean(axis=2)

    # Load image centers
    matlab_data = matlabIo.loadmat(centres_path)
    centres = matlab_data['centres']

    # Distance transform
    f = np.ones(img.shape, bool)
    for idx in xrange(centres.shape[1]):
        x = int(min(max(0, centres[0, idx]), img.shape[0] - 1))
        y = int(min(max(0, centres[1, idx]), img.shape[1] - 1))

        f[x, y] = False

    distance = distance_transform_edt(f)

    def sample_box(x, y):
        margin = int(math.floor(box_size / 2))
        return img[x-margin:x+margin+1, y-margin:y+margin+1]

    # Positive samples
    pos_samples = []
    for x in xrange(image_border_margin, distance.shape[0] - image_border_margin):
        for y in xrange(image_border_margin, distance.shape[1] - image_border_margin):
            if (distance[x][y] < positive_sample_radius):
                pos_samples.append(sample_box(x, y))

    # Negative samples random sample in a band around each cell
    neg_samples = []
    while len(pos_samples) > len(neg_samples):
        x = random.randint(image_border_margin, distance.shape[0] - image_border_margin)
        y = random.randint(image_border_margin, distance.shape[1] - image_border_margin)

        d = distance[x, y]

        if negative_sample_radius > d > positive_sample_radius:
            neg_samples.append(sample_box(x, y))

    labels = np.ones(len(pos_samples))
    labels = np.append(labels, np.zeros(len(neg_samples)))

    # Transform to np array
    samples_list = neg_samples + pos_samples
    samples = np.zeros((len(samples_list), 1, box_size, box_size))

    for idx, x in enumerate(samples):
        samples[idx, 0:, :] = samples_list[idx]

    return samples, labels, img
