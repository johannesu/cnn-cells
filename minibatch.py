import numpy as np
import math


def iterator(samples, targets, img, box_size, batchsize=100, shuffle=False):
    assert len(samples) == len(targets)
    
    def sample_box(x, y):
        margin = int(math.floor(box_size / 2))
        return img[x-margin:x+margin+1, y-margin:y+margin+1]
    
    indices = np.arange(len(targets))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(targets) - batchsize + 1, batchsize):
        batch_inputs = np.zeros((batchsize,1,box_size,box_size))
        batch_targets = np.zeros(batchsize)
        for kk in range(0, batchsize,1):
            ind = indices[start_idx+kk]
            batch_inputs[kk,0:,:] = sample_box(samples[ind][0], samples[ind][1])
            batch_targets[kk] = targets[ind]
        yield batch_inputs, batch_targets
