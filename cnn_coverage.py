from __future__ import print_function
import argparse
import numpy as np
import os
from ncoverage import NCoverage
import cv2
import time
from prettytable import PrettyTable

import keras
from keras_retinanet import models

# Set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

import utils.dataframe as df
import utils.retinanet as retinanet
import utils.augment as aug
import utils.evaluate as eval


def compare_coverage(ncdict, ncdict_aug):
    similar = 0
    for key in ncdict.keys():
        if ncdict[key] == ncdict_aug[key] == True:
            similar += 1
    
    return similar


def cnn_coverage(data_path, aug_path, weights_path):
    nc_threshold = 0.2
    iou_threshold = 0.5
    precision = 0.3

    # Model build
    # ---------------------------------------------------------------------------------
    keras.backend.tensorflow_backend.set_session(retinanet.get_session())
    model = models.load_model(weights_path, backbone_name='resnet50')
    # print(model.summary())

    nc = NCoverage(model, nc_threshold)

    # Extract data
    # ---------------------------------------------------------------------------------
    gt_dict = df.read(data_path)
    
    sum_tp, sum_fp, sum_fn = 0, 0, 0
    
    # Create table for illustating results
    results = PrettyTable(['Edit method', 'Precision', 'Recall', 'Neuron coverage', 'Coverage similarity', 'Increase coverage'])
    i = 0
   
    try:
        for img, data in gt_dict.items():
            print("\n---New iteration. Image: ", img, "---")
            image_path = os.path.join(data_path, img)
            image, draw, scale = retinanet.load_image(image_path)

            # Predict
            # --------------------------------------------------------------------------------
            start = time.time()
            bboxes, scores, labels = model.predict(np.expand_dims(image, axis=0))
            print("processing time: ", time.time() - start)
            bboxes /= scale
            
            # Format boundingboxes as {object, [[x1,y1,x2,y2],[x1,y1,x2,y2],...]}
            gt_boxes = df.extract_gt(data)
            pred_boxes = retinanet.format_pred_bb(bboxes, scores, labels, precision)
            
            # ??????????????? False negatives is handeled wrong ????????????
            prediction, recall, pn_classification = eval.calculate_precision_recall(pred_boxes, gt_boxes, iou_threshold)
            sum_tp = pn_classification['true_pos']
            sum_fp = pn_classification['false_pos']
            sum_fn = pn_classification['false_neg']

            # Display detections
            # image = retinanet.insert_detections(draw, bboxes, scores, labels, 0.3)
            # retinanet.display_image(image)


            # Evaluate NC
            # --------------------------------------------------------------------------------
            print("\n---Update neuron coverage---")
            ncdict = nc.update_coverage(np.expand_dims(image, axis=0))
            covered, total, p = nc.curr_neuron_cov()


            # Add results
            results.add_row([img, precision, recall, p, '-', '-'])

            
            for aug_dir, _, aug_files in os.walk(aug_path): # returns (subdir, dirs (list), files (list))
                if aug_dir != aug_path and img in aug_files:
                    print("\n---Analysing augmented picture,", os.path.basename(aug_dir), ", of", img, "---")

                    image_path = os.path.join(aug_dir, img)
                    aug_image, draw, scale = retinanet.load_image(image_path)


                    # Predict
                    # --------------------------------------------------------------------------------
                    start = time.time()
                    bboxes, scores, labels = model.predict(np.expand_dims(aug_image, axis=0))
                    print("processing time: ", time.time() - start)
                    bboxes /= scale

                    # Format boundingboxes as {object, [[x1,y1,x2,y2],[x1,y1,x2,y2],...]}
                    gt_boxes = df.extract_gt(data)
                    pred_boxes = retinanet.format_pred_bb(bboxes, scores, labels, precision)
                    
                    # ??????????????? False negatives is handeled wrong ????????????
                    prediction, recall, pn_classification = eval.calculate_precision_recall(pred_boxes, gt_boxes, iou_threshold)

                    # Display detections
                    # aug_image = retinanet.insert_detections(draw, bboxes, scores, labels, 0.3)
                    # retinanet.display_image(aug_image)


                    # Evaluate NC
                    # --------------------------------------------------------------------------------
                    # nc_aug = nc
                    if nc.is_testcase_increase_coverage(np.expand_dims(aug_image, axis=0)):
                        print("Augmented image increases coverage and should be added to population.")
                        increase = True
                    else: increase = False

                    print("\n---Update neuron coverage with augmented image---")
                    nc_aug = NCoverage(model, nc_threshold)
                    ncdict_aug = nc_aug.update_coverage(np.expand_dims(aug_image, axis=0)) # This may be done on a new nc object
                    covered, total, p = nc_aug.curr_neuron_cov()

                    similar_neurons = compare_coverage(ncdict, ncdict_aug)
                    similarity = similar_neurons / total

                    
                    results.add_row([os.path.basename(aug_dir), precision, recall, p, similarity, increase])


            i += 1
            if i <= len(gt_dict):
                print("\n---New image. Reset coverage dictionary---")
                nc.reset_cov_dict() 

    except KeyboardInterrupt:
        pass

    precision_total = eval.calculate_precision(sum_tp, sum_fp, sum_fn)
    recall_total = eval.calculate_recall(sum_tp, sum_fp, sum_fn)
    results.add_row(["TOTAL: ", precision_total, recall_total, '-', '-', '-'])

    # Display result for all iterations
    print("\n", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/data/data',
                        help='Path for dataset.')
    parser.add_argument('--weights', type=str, default=os.path.join(os.pardir, 'keras-retinanet/snapshots', 'resnet50_csv_04_inference.h5'),
                        help='Path for weights. Should be added to ../keras-retinanet/snapshots')
    args = parser.parse_args()


    print("\n---Augmenting---")
    aug_save = os.path.join(args.dataset, os.pardir, "augmented")
    if not os.path.exists(aug_save):
        os.makedirs(aug_save)  

    aug.augment(args.dataset, aug_save)

    cnn_coverage(args.dataset, aug_save, args.weights)
