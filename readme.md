# CNN cell detection

This code show how to train a cell detector using a convolutional neural network in [Lasagne](https://github.com/Lasagne/Lasagne).

### Getting started

Look at [main.ipynb](main.ipynb).

#### Requirements

* Python 2 or 3
* The python packages in [requirements.txt](requirements.txt), if you have pip you can install them using:

```shell
pip3 install -r requirements.txt
```

### Details
* Each image is manually annotated with the center point of each cell as well as some hard negative examples
* All points within `sample radius` of a cell centre are sampled as `positive samples`
* An equal number of `negative samples` are randomly sampled outside the `positive radius`
* All points within `sample radius` of the hard negative examples are sampled as `negative samples`
* A [convolutional neural network](network.py) is trained using the negative and positive samples. For each sample, a box of size `box_size`, is used as input to the network.
* Given a new image a `box_sized` window is slided through each possible patch in the image, generating a probability map
* Local maxima in the probability map are marked as cell centers

Note: There is no padding on the boundary so no detection is possible `box_size/2` pixels from the image boundary.

![Description](images/description.png)

#### Credit
The network and code structure is based on Lasanges `MNIST` example
[https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py](https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py)
