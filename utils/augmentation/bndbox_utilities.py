import xml.etree.ElementTree as ElementTree
import numpy as np
import cv2
from utils.dataframe import XmlDictConfig


def get_bndbox(tree):
    root = tree.getroot()
    xmldict = XmlDictConfig(root)
    bndbox = []
    if type(xmldict['object']) == type([]):
        for elem in xmldict['object']:
            xmin,ymin,xmax,ymax = elem['bndbox']['xmin'],elem['bndbox']['ymin'],elem['bndbox']['xmax'],elem['bndbox']['ymax']
            bndbox.append((int(xmin), int(ymin), int(xmax), int(ymax)))
    else:
        xmin,ymin,xmax,ymax = xmldict['object']['bndbox']['xmin'],xmldict['object']['bndbox']['ymin'],xmldict['object']['bndbox']['xmax'],xmldict['object']['bndbox']['ymax']
        bndbox.append((int(xmin), int(ymin), int(xmax), int(ymax)))
    return bndbox

def update_bndbox(tree, bndbox):
    root = tree.getroot()
    object_count = len(bndbox)
    for i in range(object_count):
        xmin, ymin, xmax, ymax = bndbox[i]
        root[6+i][4][0].text = str(xmin)
        root[6+i][4][1].text = str(ymin)
        root[6+i][4][2].text = str(xmax)
        root[6+i][4][3].text = str(ymax)

def draw_bndbox(img, xmin, ymin, xmax, ymax):
    cv2.rectangle(img=img, pt1=(xmin,ymin), pt2=(xmax,ymax), color=(0,0,255), thickness=1)

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
    bndbox = get_bndbox(tree)
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
