import pickle
import os
import numpy as np
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

def visualize(np_arr):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', np_arr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_batch(batch='data_batch_1'):
    dict = unpickle(file=os.path.join('cifar-10-batches-py',batch))
    data = dict[b'data']
    mask = np.asarray(dict[b'labels']) == 8
    filtered_data = data[mask]
    num_images = filtered_data.shape[0]
    cifar10 = np.zeros((32,32,3,num_images), dtype=np.float32)
    for i in range(num_images):
        raw = filtered_data[i,:]
        img = raw.reshape((32,32,3),order='F')
        img = np.rot90(img, 3)
        cifar10[...,i] = img
    #visualize(cifar10[...,0]/255)
    print('Cifar10 loaded')
    return cifar10

if __name__ ==  '__main__':
    dict = unpickle(file=os.path.join('cifar-10-batches-py','data_batch_1'))
    #Create a mask to filter on class:
    data = dict[b'data']
    print(data.shape)
    mask = np.asarray(dict[b'labels']) == 8
    filtered_data = data[mask]
    print(filtered_data.shape)
    load_cifar10_batch()
