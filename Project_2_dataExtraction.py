"""
@author: Diego 
"""

"""
this is the code you need to run to import data.
You may have to change line 36 selecting the correct path.
"""
import os
import gzip
import numpy as np

def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


cwd = '../'

X_all_labels, y_all_labels = load_mnist(cwd, kind='train')

"""
We are only interested in the items with label 2, 4 and 6.
Only a subset of 1000 samples per class will be used.
"""
indexLabel3 = np.where((y_all_labels==3))
xLabel3 =  X_all_labels[indexLabel3][:1000,:].astype('float64')
yLabel3 = y_all_labels[indexLabel3][:1000].astype('float64')

indexLabel8 = np.where((y_all_labels==8))
xLabel8 =  X_all_labels[indexLabel8][:1000,:].astype('float64')
yLabel8 = y_all_labels[indexLabel8][:1000].astype('float64')

indexLabel6 = np.where((y_all_labels==6))
xLabel6 =  X_all_labels[indexLabel6][:1000,:].astype('float64')
yLabel6 = y_all_labels[indexLabel6][:1000].astype('float64')


"""
To train a SVM in case of binary classification you have to convert the labels of the two classes of interest into '+1' and '-1'.
"""