import xml.etree.ElementTree as ElementTree
import numpy as np
import cv2
# from xmlDictConfig import XmlDictConfig
from utils.dataframe import XmlDictConfig


def get_bndbox(tree, image_size=None):
    root = tree.getroot()
    xmldict = XmlDictConfig(root)
    bndbox = []
    classes = []
    if type(xmldict['object']) == type([]):
        for elem in xmldict['object']:
            xmin,ymin,xmax,ymax = elem['bndbox']['xmin'],elem['bndbox']['ymin'],elem['bndbox']['xmax'],elem['bndbox']['ymax']
            class_name = elem['name']
            bndbox.append((int(xmin), int(ymin), int(xmax), int(ymax)))
            classes.append(class_name)
    else:
        xmin,ymin,xmax,ymax = xmldict['object']['bndbox']['xmin'],xmldict['object']['bndbox']['ymin'],xmldict['object']['bndbox']['xmax'],xmldict['object']['bndbox']['ymax']
        class_name = xmldict['object']['name']
        bndbox.append((int(xmin), int(ymin), int(xmax), int(ymax)))
        classes.append(class_name)

    if image_size is not None:
        scaled_boxes = []
        for box in bndbox:
            scaled_boxes.append(scale_bndbox(tree=tree, bndbox=box, image_size=image_size))
        return scaled_boxes, classes
    else:
        return bndbox, classes

def update_bndbox(tree, bndbox):
    root = tree.getroot()
    object_count = len(bndbox)
    for i in range(object_count):
        xmin, ymin, xmax, ymax = bndbox[i]
        root[6+i][4][0].text = str(xmin)
        root[6+i][4][1].text = str(ymin)
        root[6+i][4][2].text = str(xmax)
        root[6+i][4][3].text = str(ymax)

def draw_bndbox(img, xmin, ymin, xmax, ymax, cls, color):
    cv2.rectangle(img=img, pt1=(xmin,ymin), pt2=(xmax,ymax), color=color, thickness=1)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (int(xmin),int(ymin-2))
    fontScale              = 0.2
    fontColor              = (255,255,255)
    lineType               = 1

    if cls == 1:
        label = '1'
    elif cls == 2:
        label = '2'
    elif cls == 3:
        label = '3'
    elif cls == 4:
        label = '4'
    cv2.putText(img, label, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

def iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    intersection = max(0, x_b - x_a) * max(0,y_b - y_a)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return intersection / float(area_a + area_b - intersection)

def rotate_bounding_box(tree, M):
    bndbox,_ = get_bndbox(tree, )
    new_bndbox = []
    for elem in bndbox:
        xmin, ymin, xmax, ymax = elem

        x1,y1 = xmin, ymin
        x2,y2 = xmin, ymax
        x3,y3 = xmax, ymax
        x4,y4 = xmax, ymin

        corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
        corners = corners.reshape(-1,2)
        corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype=type(corners[0][0]))))

        calculated = np.dot(M,corners.T).T
        calculated = calculated.reshape(-1,8)

        x_ = calculated[:,[0,2,4,6]]
        y_ = calculated[:,[1,3,5,7]]

        xmin_new = int(np.min(x_,1).reshape(-1,1))
        ymin_new = int(np.min(y_,1).reshape(-1,1))
        xmax_new = int(np.max(x_,1).reshape(-1,1))
        ymax_new = int(np.max(y_,1).reshape(-1,1))

        new_bndbox.append((xmin_new, ymin_new, xmax_new, ymax_new))

    update_bndbox(tree, new_bndbox)

def scale_bndbox(tree, bndbox, image_size):
    img_h, img_w = image_size
    root = tree.getroot()
    current_w = int(root[4][0].text)
    current_h = int(root[4][1].text)
    current_channels = int(root[4][2].text)

    ratio_w = img_w / current_w
    ratio_h = img_h / current_h
    #bndboxes = get_bndbox(tree)
    #new_bndbox = []
    xmin, ymin, xmax, ymax = bndbox
    scaled_xmin = xmin*ratio_w
    scaled_xmax = xmax*ratio_w
    scaled_ymin = ymin*ratio_h
    scaled_ymax = ymax*ratio_h
    #gt_box = [int(i) for i in gt_box]
    #new_bndbox.append((scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax))
    return [int(scaled_xmin), int(scaled_ymin), int(scaled_xmax), int(scaled_ymax)]
    #update_bndbox(tree, new_bndbox)
