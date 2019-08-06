import xml.etree.ElementTree as ElementTree
import sys
import os
import numpy as np
import pandas as pd
import csv
import pandas



class XmlDictConfig(dict):
    def __init__(self, parent_element):
        childrenNames = []
        for child in parent_element.getchildren():
            childrenNames.append(child.tag)

        if parent_element.items(): #attributes
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                #print len(element), element[0].tag, element[1].tag
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))

                if childrenNames.count(element.tag) > 1:
                    try:
                        currentValue = self[element.tag]
                        currentValue.append(aDict)
                        self.update({element.tag: currentValue})
                    except: #the first of its kind, an empty list must be created
                        self.update({element.tag: [aDict]}) #aDict is written in [], i.e. it will be a list

                else:
                     self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})


def read(inputDirectory):
    df = {}

    for path, dirnames, filenames in os.walk(inputDirectory):
        for name in filenames:
            if '.xml' in name and (name[:-4] + '.jpg') in filenames:
                tree = ElementTree.parse(os.path.join(path, name))
                root = tree.getroot()
                xmldict = XmlDictConfig(root)
                df[xmldict['filename']] = xmldict['object']

    return df



def extract_gt(dict):
    """ 
    # returns
        df     : A dictionary with value as a list of 4 elements (x1, y1, x2, y2).
    """ 
    df = {}

    if type(dict) == type([]): # Multiple values/detections
        for key in dict:
            if key['name'] in df:
                df[key['name']] = np.append(df[key['name']], [[int(key['bndbox']['xmin']), 
                                                                int(key['bndbox']['ymin']), 
                                                                int(key['bndbox']['xmax']), 
                                                                int(key['bndbox']['ymax'])]], axis=0)
            else:
                df[key['name']] = np.array([[int(key['bndbox']['xmin']), 
                                        int(key['bndbox']['ymin']), 
                                        int(key['bndbox']['xmax']), 
                                        int(key['bndbox']['ymax'])]])
              
    else:
        df[dict['name']] = np.array([[int(dict['bndbox']['xmin']), 
                                    int(dict['bndbox']['ymin']), 
                                    int(dict['bndbox']['xmax']), 
                                    int(dict['bndbox']['ymax'])]])

    return df



def format_pred_bb(boxes, scores, labels, labels_to_names, precision=0.5):
    """ 
    # Arguments
        boxes     : A list of 4 elements (x1, y1, x2, y2).
    """    
    detections = {}
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < precision:
            break
            
        if labels_to_names[label] in detections:
            detections[labels_to_names[label]] = np.append(detections[labels_to_names[label]], [[box[0], box[1], box[2], box[3]]], axis=0) 
        else:
            detections[labels_to_names[label]] = np.array([[box[0], box[1], box[2], box[3]]])

    return detections



def read_classes(csv_path):
    classes = {}
    file = open(csv_path, "r")
    for line in file:
        data = line[:-1].split(',')
        classes[int(data[1])] = data[0]

    return classes



def exists(csv_name, image_name):
    index = 1 # CSV is one-indexed
    total_tp, total_fp, total_fn = 0, 0, 0

    with open(csv_name, 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            total_tp += int(row[1])
            total_fp += int(row[2])
            total_fn += int(row[3])
            if image_name in row[0]:
                return True, index + 1, total_tp, total_fp, total_fn
            index += 1

    return False, index, total_tp, total_fp, total_fn