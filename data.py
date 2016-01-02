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

    # Distance transform
    f = np.ones(img.shape, bool)
    for idx in xrange(centres.shape[1]):
        x = int(min(max(0, centres[0, idx]), img.shape[0] - 1))
        y = int(min(max(0, centres[1, idx]), img.shape[1] - 1))

        f[x, y] = False

    distance = distance_transform_edt(f)

    # Positive samples within radius positive_sample radius from a cell centre.
    samples_list = []
    for x in xrange(image_border_margin, distance.shape[0] - image_border_margin):
        for y in xrange(image_border_margin, distance.shape[1] - image_border_margin):
            if (distance[x][y] < positive_sample_radius):
                samples_list.append((x, y))

    # Negative samples random sample in a band around each cell
    nbr_positives = len(samples_list)
        
    while len(samples_list) < 2*nbr_positives:
        x = random.randint(image_border_margin, distance.shape[0] - image_border_margin)
        y = random.randint(image_border_margin, distance.shape[1] - image_border_margin)

        d = distance[x, y]

        if negative_sample_radius > d > positive_sample_radius:
            samples_list.append((x, y))

	# Load negative samples
    matlab_data = matlabIo.loadmat(neg_path)
    negative_centres = matlab_data['samples']

    # Distance transform
    f = np.ones(img.shape, bool)
    for idx in xrange(negative_centres.shape[1]):
        x = int(min(max(0, negative_centres[0, idx]), img.shape[0] - 1))
        y = int(min(max(0, negative_centres[1, idx]), img.shape[1] - 1))

        f[x, y] = False

    distance = distance_transform_edt(f)

    # Negative samples within radius positive_sample radius from a negative centres.
    for x in xrange(image_border_margin, distance.shape[0] - image_border_margin):
        for y in xrange(image_border_margin, distance.shape[1] - image_border_margin):
            if (distance[x][y] < positive_sample_radius):
                samples_list.append((x, y))

	# Fill in the ground truth labels
	labels = np.ones(nbr_positives)
	labels = np.append(labels, np.zeros(len(samples_list)-nbr_positives))

    # Transform to np array
    samples = np.zeros((len(samples_list), 2))

    for idx, x in enumerate(samples):
        samples[idx,0] = samples_list[idx][0]
        samples[idx,1] = samples_list[idx][1]

    return samples, labels, img
