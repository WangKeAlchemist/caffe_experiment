__author__ = 'ke-wang'

import numpy as np
import caffe
import matplotlib as plt
from caffe import layers as L
from caffe import params as P
import os

def logreg(hdf5, batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
    # n.ip1 = L.InnerProduct(n.data,)

os.system('~/caffe/build/tools/caffe train blabla')