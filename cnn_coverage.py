from __future__ import print_function
import argparse
import numpy as np
import os
#from scipy.misc import imread, imresize
from imageio import imread
#import Image # resize
from keras import backend as K
from ncoverage import NCoverage
import csv
import cv2
import time
from prettytable import PrettyTable

import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# Set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

import dataframe as df
import retinanet
import augment as aug
import evaluate as eval

def preprocess_input_InceptionV3(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def exact_output(y):
    return y

def normalize_input(x):
    return x / 255.

def read_image(image_file, image_size):

    img = imread(image_file)
        # Cropping
    crop_img = img[200:, :]
        # Resizing
    img = imresize(crop_img, size=image_size)
    imgs = []
    imgs.append(img)
    if len(imgs) < 1:
        print('Error no image at timestamp')

    img_block = np.stack(imgs, axis=0)
    if K.image_dim_ordering() == 'th':
        img_block = np.transpose(img_block, axes=(0, 3, 1, 2))
    return img_block

def read_images(seed_inputs, seed_labels, image_size):

    img_blocks = []
    for file in os.listdir(seed_inputs):
        if file.endswith(".jpg"):
            img_block = read_image(os.path.join(seed_inputs, file), image_size)
            img_blocks.append(img_block)
    return img_blocks


def compare_coverage(ndict_original, ndict_augmented):
    similar = 0
    for key in ndict_original.keys():
        if ndict_original[key] == ndict_augmented[key] == True:
            similar += 1
    
    return similar


def cnn_coverage(data_path, csv_path):
    nc_threshold = 0.2
    iou_threshold = 0.5
    precision = 0.3
    weights_path = "/home/hellum/DL/keras-retinanet/snapshots/resnet50_csv_04_inference.h5" # './weights_HMB_2.hdf5' # Change to your model weights

    # Model build
    # ---------------------------------------------------------------------------------
    keras.backend.tensorflow_backend.set_session(retinanet.get_session())
    model = models.load_model(weights_path, backbone_name='resnet50')
    # print(model.summary())

    nc = NCoverage(model, nc_threshold)

    # Extract data
    # ---------------------------------------------------------------------------------
    dict = df.read(data_path)

    ###############################################################################################################

    # PSEUDO: 
    # for imagepath, gt in dictionary:

    #     predict on imagepath

    #     convert gt to numpy format
    #     calculate true positive, false positive, true negative and false negative
    #     calculate precision and recall 

    #     calculate neuron coverage on prediction
    #     save coverage in dictionary

    #     augment image

    #     for augmentedimage in augmentedimages
    #         predict on augmentedimage

    #         calculate true positive, false positive, true negative and false negative
    #         calculate precision and recall 

    #         calculate neuron coverage on prediction
    #         save coverage in dictionary

        
    #     display and compare results:
    #             name    |   precision   |   recall  |   neuron coverage | Percentage of similar neurons being fired
    #         ------------|---------------|-----------|-------------------|-------------------------------------------
    #              .      |               |           |                   |
    #              .      |               |           |                   |
    #              .      |               |           |                   |


    ################################################################################################################


    sum_tp, sum_fp, sum_fn = 0, 0, 0
    
    # Create table for illustating results
    results = PrettyTable(['Edit method', 'Precision', 'Recall', 'Neuron coverage'])

    for img, data in dict.items():
        image_path = os.path.join(data_path, img)
        # image_path = "/home/hellum/Pictures/boats_multiple.jpg"
        image, draw, scale = retinanet.load_image(image_path)

        # Predict
        # --------------------------------------------------------------------------------
        
        # Process image using model    
        start = time.time()
        bboxes, scores, labels = model.predict(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)
                
        # Correct for image scale
        bboxes /= scale

        print("\n---Update neuron coverage---")
        ndict_original = nc.update_coverage(np.expand_dims(image, axis=0))
        covered, total, p = nc.curr_neuron_cov()

        # KAN DETTE BRUKES FOR Å FINNE ALLE AKTIVERTE NEVRONER OG SAMMENLIGNE???
        ####################################################################################
        # tempk = []
        # for k in ndict_original.keys():
        #     if ndict_original[k] == True:
        #         tempk.append(k)
        # tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
        # covered_detail = ';'.join(str(x) for x in tempk).replace(',', ':')
        # ####################################################################################

        print("\n---Covered neurons---")
        covered_neurons = nc.get_neuron_coverage(np.expand_dims(image, axis=0))
        print('input covered {} neurons'.format(covered_neurons))

        print("\n---Reset coverage dictionary---")
        nc.reset_cov_dict() 

        # Display detections
        # image = retinanet.insert_detections(draw, bboxes, scores, labels, 0.3)
        # retinanet.display_image(image)


        # Format boundingboxes as {object, [[x1,y1,x2,y2],[x1,y1,x2,y2],...]}
        gt_boxes = df.extract_gt(data)
        pred_boxes = retinanet.format_pred_bb(bboxes, scores, labels, precision)
        
        # ??????????????? False negatives is handeled wrong ????????????
        prediction, recall, pn_classification = eval.calculate_precision_recall(pred_boxes, gt_boxes, sum_tp, sum_fp, sum_fn, iou_threshold)
        sum_tp = pn_classification['true_pos']
        sum_fp = pn_classification['false_pos']
        sum_fn = pn_classification['false_neg']

        
        # Add results
        results.add_row([img, precision, recall, p])

        # Augment
        # --------------------------------------------------------------------------------
        # for image in aug.apply(image) # Apply augmentation and return all images in some sort (THIS IS NOT YET IMPLEMENTED)    
            # do the same as for original pictures

            # similar_neurons = compare_coverage(ndict_original, ndict_augmented)
            # similarity = similar_neurons / total
    # Display result for all iterations
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/home/hellum/DL/keras-retinanet/images/Dataset_sample',
                        help='path for dataset')
    parser.add_argument('--csv', type=str, default='/home/hellum/DL/keras-retinanet/images/csv/all.csv',
                        help='path for dataset')
    # parser.add_argument('--index', type=int, default=0,
    #                     help='different indice mapped to different transformations and params')
    args = parser.parse_args()
    # epoch_testgen_coverage(args.index, args.dataset)
    cnn_coverage(args.dataset, args.csv)
