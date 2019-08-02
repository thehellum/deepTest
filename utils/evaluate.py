import numpy as np
import matplotlib.pyplot as plt
import json
# import copy

def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.
    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
        returns:
            float: value of the intersection of union for the two boxes.
    """
    
    area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    area_prediction = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
    intersection_x = min(gt_box[2],prediction_box[2]) - max(gt_box[0],prediction_box[0])
    intersection_y = min(gt_box[3],prediction_box[3]) - max(gt_box[1],prediction_box[1])
    if intersection_x > 0 and intersection_y > 0:
        intersection = intersection_x * intersection_y
        union = area_gt + area_prediction - intersection
        iou = (float(intersection) / union)
    else:
        iou = 0
    return iou

def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if (num_tp+num_fp) == 0:
        return 1

    return num_tp/(num_tp+num_fp)



def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if (num_tp+num_fn) == 0:
        return 0

    return num_tp/(num_tp+num_fn)



def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.
        Remember: Matching of bounding boxes should be done with decreasing IoU order!
    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    
    # Matching bounding boxes
    Bp_match = [] # predicted
    Ba_match = [] # actual (ground truth)
    n = 0
    # Find all possible matches with a IoU >= iou threshold
    # Sort all matches on IoU in descending order
    # Find all matches with the highest IoU threshold
    for Ba in gt_boxes:
        IoU_max = -1
        Bp_max = []
        for Bp in prediction_boxes:
            IoU = calculate_iou(Bp, Ba)
            if (IoU >= iou_threshold and IoU > IoU_max):
                IoU_max = IoU
                Bp_max = Bp

       
        if IoU_max != -1:
            Bp_match.append(Bp_max)
            Ba_match.append(Ba)

    Bp_match = np.asarray(Bp_match)
    Ba_match = np.asarray(Ba_match)

    return Bp_match, Ba_match



def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!
    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, "false_neg": int}
    """
    # Find the bounding box matches with the highes IoU threshold

    Bp_match, Ba_match = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    num_tp = Ba_match.shape[0]
    num_fp = prediction_boxes.shape[0] - Bp_match.shape[0]
    num_fn = gt_boxes.shape[0] - Ba_match.shape[0]

    return {"true_pos": num_tp, "false_pos": num_fp, "false_neg": num_fn}, Bp_match, Ba_match



def calculate_precision_recall(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Given a set of prediction boxes and ground truth boxes for single images,
       calculates recall and precision over single images.
       
       NB: all_prediction_boxes and all_gt_boxes are not matched!
    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        pred_boxes: A dictionary with value as a list of 4 elements (x1, y1, x2, y2).
        gt_boxes: A dictionary with value as a list of 4 elements (x1, y1, x2, y2).
    Returns:
        tuple: (precision, recall). Both float.
    """ 
    sum_tp = 0
    sum_fp = 0
    sum_fn = 0

    for obj_gt, labels_gt in gt_boxes.items():  
        # print(obj_gt, ",", labels_gt
        for obj_pred, label_pred in pred_boxes.items():
            if obj_gt != obj_pred:
                continue

            pn_classification, Bp_match, Ba_match = calculate_individual_image_result(label_pred, labels_gt, iou_threshold)
            sum_tp += pn_classification['true_pos']
            sum_fp += pn_classification['false_pos']
            sum_fn += pn_classification['false_neg']

    precision = calculate_precision(sum_tp, sum_fp, sum_fn)
    recall = calculate_recall(sum_tp, sum_fp, sum_fn)

    return precision, recall, {"true_pos": sum_tp, "false_pos": sum_fp, "false_neg": sum_fn}



def nms_consider_label(scores, boxes, labels, score_threshold, nms_threshold):
    # Keep predictions where scores > threshold
    indices = np.concatenate(np.argwhere(scores[0] > score_threshold))
    scores = np.expand_dims(scores[0][indices], axis=0)
    boxes = np.expand_dims(boxes[0][indices], axis=0)
    labels = np.expand_dims(labels[0][indices], axis=0)
    
    # Suppress
    i = 1
    deleted = []
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        j = i
        while j < len(scores[0]):
            if (label == labels[0][j]):
                if j not in deleted: 
                    if calculate_iou(boxes[0][j], box) >= nms_threshold:
                        indices = np.delete(indices, j-len(deleted))
                        deleted.append(j)
            j += 1
        i += 1

    scores = np.expand_dims(scores[0][indices], axis=0)
    boxes = np.expand_dims(boxes[0][indices], axis=0)
    labels = np.expand_dims(labels[0][indices], axis=0)

    return boxes, scores, labels



def nms(scores, boxes, labels, score_threshold, nms_threshold):
    # Only keep predictions where scores > threshold
    try:
        indices = np.concatenate(np.argwhere(scores[0] > score_threshold))
        scores = np.expand_dims(scores[0][indices], axis=0)
        boxes = np.expand_dims(boxes[0][indices], axis=0)
        labels = np.expand_dims(labels[0][indices], axis=0)

        # Suppress
        i = 1
        deleted = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            j = i
            while j < len(scores[0]):
                if j not in deleted: 
                    if calculate_iou(boxes[0][j], box) >= nms_threshold:
                        indices = np.delete(indices, j-len(deleted))
                        deleted.append(j)
                j += 1
            i += 1

        scores = np.expand_dims(scores[0][indices], axis=0)
        boxes = np.expand_dims(boxes[0][indices], axis=0)
        labels = np.expand_dims(labels[0][indices], axis=0)
    
    except ValueError:
        scores = [[]]
        boxes = [[]]
        labels = [[]]

    return boxes, scores, labels





# Not used
######################################################################################################

def calculate_precision_recall_all_images(all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images.
       
       NB: all_prediction_boxes and all_gt_boxes are not matched!
    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """

    sum_tp = 0
    sum_fp = 0
    sum_fn = 0
   
    
    for i in range ( len(all_gt_boxes) ):
        pos_neg = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
        sum_tp += pos_neg['true_pos']
        sum_fp += pos_neg['false_pos']
        sum_fn += pos_neg['false_neg']

    precision = calculate_precision(sum_tp, sum_fp, sum_fn)
    recall = calculate_recall(sum_tp, sum_fp, sum_fn)

    return (precision, recall)

    # Find total true positives, false positives and false negatives
    # over all images

    # Compute precision, recall


def get_precision_recall_curve(all_prediction_boxes, all_gt_boxes,
                               confidence_scores, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the precision-recall curve over all images. Use the given
       confidence thresholds to find the precision-recall curve.
       NB: all_prediction_boxes and all_gt_boxes are not matched!
    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]
            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        tuple: (precision, recall). Both np.array of floats.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    # DO NOT CHANGE. If you change this, the tests will not pass when we run the final
    # evaluation
    confidence_thresholds = np.linspace(0, 1, 500)
    the_prediction = []
    recall = []
    precision = []
    for i in confidence_thresholds:
        for (num_img, image) in enumerate(all_prediction_boxes):
            prediction = []
            for (num_box, box) in enumerate(image):
                if confidence_scores[num_img][num_box] >= i:
                    prediction.append(box)
            prediction = np.asarray(prediction)
            the_prediction.append(prediction) 
        rec_pre = calculate_precision_recall_all_images(
                the_prediction, all_gt_boxes, iou_threshold)
        the_prediction = []
        recall.append(rec_pre[1])
        precision.append(rec_pre[0])

    return(np.asarray(precision), np.asarray(recall))
  


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'
    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    # No need to edit this code.
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")
    

def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.
    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    # DO NOT CHANGE. If you change this, the tests will not pass when we run the final
    # evaluation
    recall_levels = np.linspace(0, 1.0, 11)
    values = []
    for it in recall_levels:
        max_p = 0
        for (precision, recall) in zip(precisions,recalls):
            if recall >= it and precision > max_p:
                max_p = precision 
        values.append(max_p)
 
    AP = (1/len(recall_levels)) * (np.sum(values))

    return AP

def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5
    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)
    iou_threshold = 0.5
    precisions, recalls = get_precision_recall_curve(all_prediction_boxes,
                                                     all_gt_boxes,
                                                     confidence_scores,
                                                     iou_threshold)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions,
                                                              recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))
