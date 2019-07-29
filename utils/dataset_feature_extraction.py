import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import statistics as stat
from utils.imageLoader import ImageLoader
from utils.read_cifar10 import load_cifar10_batch
from random import shuffle

def visualize(np_arr):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', np_arr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_image_intensity(img, grid=(1,1)):
    h,w,channels = img.shape
    h_shift = h//grid[0]
    w_shift = w//grid[1]
    grid_mean = np.zeros(shape=(grid[0],grid[1],channels))

    for i in range(grid[0]):
        for j in range(grid[1]):
            for k in range(channels):
                patch = img[i*(h_shift):(i+1)*(h_shift),(j)*w_shift:(j+1)*w_shift,k]
                grid_mean[i,j,k] = np.mean(a=patch, axis=(0,1))
    return grid_mean.reshape((1,grid[0]*grid[1]*channels))

def get_image_variance(img, grid=(1,1)):
    h,w,channels = img.shape
    h_shift = h//grid[0]
    w_shift = w//grid[1]
    grid_var = np.zeros(shape=(grid[0],grid[1],channels))

    for i in range(grid[0]):
        for j in range(grid[1]):
            for k in range(channels):
                patch = img[i*(h_shift):(i+1)*(h_shift),(j)*w_shift:(j+1)*w_shift,k]
                grid_var[i,j,k] = np.var(a=patch, axis=(0,1))
    return grid_var.reshape((1,grid[0]*grid[1]*channels))

def extract_color_features(imgs, grid=(1,1)):
    feature_matrix = np.zeros(shape=(imgs.index, grid[0]*grid[1]*3*2))
    for i in range(imgs.index):
        img = imgs.database[...,i]
        grid_mean = get_image_intensity(img, grid=grid)
        grid_var = get_image_variance(img, grid=grid)
        feature_vector = np.concatenate(seq=(grid_mean,grid_var), axis=1)
        feature_matrix[i,:] = feature_vector
    return feature_matrix


def compute_dataset_intensity(imgs, grid=(1,1)):
    lowest_intensity = 1.0
    lowest_intensity_index = -1
    highest_intensity = 0.0
    highest_intensity_index = -1
    intensities = np.zeros(shape=(grid[0], grid[1], imgs.index))
    for i in range(imgs.index):
        print(i)
        intensities[...,i] = get_image_intensity(img=imgs.database[...,i], grid=grid)
    print('Highest intensity: ', max(intensities.flatten()))
    print('Lowest intensity: ', min(intensities.flatten()))
    print(np.where(intensities == max(intensities.flatten())))
    print(np.where(intensities == min(intensities.flatten())))
    #if grid == (3,3):
    #    intensities_split = intensities[1:2,0:3,:]
    #    return intensities_split.flatten()
    return intensities.flatten()

def plot_histogram(data):
    print(len(data))
    if len(data) == 1:
        n, bins, patches = plt.hist(data[0], bins=50, range=(0,255), normed=1, facecolor='green', alpha=0.75)
        plt.grid()
    else:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False)
        ax1.hist(data[0], bins=50, range=(0,255), facecolor='red', alpha=0.75)
        ax2.hist(data[1], bins=50, range=(0,255), facecolor='green', alpha=0.75)
        ax3.hist(data[2], bins=50, range=(0,255), facecolor='blue', alpha=0.75)
    plt.show()

def gaussian_filtering(imgs):
    for i in range(imgs.database.shape[-1]):
        imgs.database[...,i] = cv2.GaussianBlur(src=imgs.database[...,i], ksize=(15,15), sigmaX=5, sigmaY=5)

def rgb_2_gray(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def pixel_mean(imgs):
    #print(np.mean(a=imgs.database, axis=-1))
    mean = np.var(a=imgs.database, axis=-1)
    diff = mean.max() - mean.min()
    mean = mean - mean.min()
    mean = (255 * mean)/ diff
    mean = np.uint8(mean)
    return mean

def get_pixel_value(patch,x,y,center):
    if patch[x,y] >= center:
        return 1
    elif patch[x,y] < center:
        return 0
    else:
        print('get_pixel_value() error')

def compute_local_binary_pattern(patch):
    # Assume grayscaled image
    h,w = patch.shape
    values = np.zeros(shape=(h,w))
    patch_padded = np.pad(array=patch, pad_width=((1,1),(1,1)), mode='constant', constant_values=((0,0),(0,0)))
    for x in range(1,h+1):
        for y in range(1,w+1):
            center = patch_padded[x,y]
            values[x-1,y-1] += get_pixel_value(patch_padded,x-1,y+1,center) * 1 # Upper right corner
            values[x-1,y-1] += get_pixel_value(patch_padded,x,y+1,center) * 2
            values[x-1,y-1] += get_pixel_value(patch_padded,x+1,y+1,center) * 4
            values[x-1,y-1] += get_pixel_value(patch_padded,x+1,y,center) * 8
            values[x-1,y-1] += get_pixel_value(patch_padded,x+1,y-1,center) * 16
            values[x-1,y-1] += get_pixel_value(patch_padded,x,y-1,center) * 32
            values[x-1,y-1] += get_pixel_value(patch_padded,x-1,y-1,center) * 64
            values[x-1,y-1] += get_pixel_value(patch_padded,x-1,y,center) * 128
    hist, bin_edges = np.histogram(a=values, bins=256, range=(0,255))
    ubp_hist = uniform_binary_pattern(hist)
    ubp_hist = ubp_hist / ubp_hist.sum()
    return ubp_hist

def uniform_binary_pattern(hist):
    binary_patterns = [0,1,2,3,4,6,7,8,12,14,15,16,24,28,30,31,32,
                       48,56,60,62,63,64,96,112,120,124,126,127,128,
                       129,131,135,143,159,191,192,193,195,199,207,
                       223,224,225,227,231,239,240,241,243,247,248,
                       249,251,252,253,254,255]
    uniform_binary_pattern_hist = np.zeros(shape=(1,59))
    index = 0
    for i in range(len(hist)):
        if i in binary_patterns:
            uniform_binary_pattern_hist[:,index] = hist[i]
            index += 1
        elif i not in binary_patterns:
            uniform_binary_pattern_hist[:,-1] += hist[i]
    return uniform_binary_pattern_hist

def get_spatial_feature_vector(img, grid=(1,1)):
    h,w = img.shape
    h_shift = h//grid[0]
    w_shift = w//grid[1]
    feature_vector = np.zeros(shape=(grid[0],grid[1],59))
    for i in range(grid[0]):
        for j in range(grid[1]):
            patch = img[i*(h_shift):(i+1)*(h_shift),(j)*w_shift:(j+1)*w_shift]
            feature_vector[i,j,:] = compute_local_binary_pattern(patch)
    return feature_vector.reshape((1,grid[0]*grid[1]*59))

def extract_spatial_features(imgs, grid=(1,1)):
    feature_matrix = np.zeros(shape=(imgs.index, grid[0]*grid[1]*59))
    for i in range(imgs.index):
        print('Features extracted from image ', i)
        img = imgs.database[...,i]
        feature_vector = get_spatial_feature_vector(img, grid)
        feature_matrix[i,:] = feature_vector
    return feature_matrix

if __name__ == "__main__":
    imgs_train = ImageLoader(shape=(100,100), scale=1)
    imgs_train.load_from_file(file_path='Datasets_np/full_rgb_scaled1_100x100_shuffled.npy')
    imgs_train_gray = ImageLoader(shape=(100,100), scale=1, mode='gray', size=100)
    imgs_train_gray.load_from_dir(dir_path='Dataset_full')
    extract_color_features(imgs_train, grid=(3,3))
    extract_spatial_features(imgs_train_gray, grid=(3,3))
    # numpy.cov
    # numpy.corrcoef
