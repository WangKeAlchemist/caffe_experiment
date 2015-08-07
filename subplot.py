__author__ = 'ke-wang'


import sys
import numpy as np
import Image
caffe_root = '/home/ke-wang/caffe/'
sys.path.insert(0,caffe_root+'python')
import caffe
from pylab import *
import matplotlib.pyplot as plt
from caffe import layers as L
from caffe import params as P

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe.set_mode_cpu()
# Load the net, list its data and params, and filter an example image.
net = caffe.Net(caffe_root+'examples/net_surgery/conv.prototxt', caffe.TEST)
print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
im = np.array(Image.open(caffe_root+'examples/images/cat_gray.jpg'))
plt.title('original image')
plt.imshow(im)
plt.axis('off')
# plt.

im_input = im[np.newaxis, np.newaxis, :, :]
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input
