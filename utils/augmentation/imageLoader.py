import cv2
import numpy as np
import sys
import os
from random import shuffle
#from xmlDictConfig import XmlDictConfig
import xml.etree.ElementTree as ElementTree

class ImageLoader:
    def __init__(self, shape=(720,1280), scale=1, mode='gray', size=100, index_start=0, has_xml=True):
        h = int(shape[0] * scale)
        w = int(shape[1] * scale)
        self.img_shape = (h,w)
        self.size = size
        self.mode = mode
        self.index = 0
        self.index_start = index_start
        self.true_index = self.index - self.index_start
        self.channels = None
        self.has_xml = has_xml
        self.xml_files = {}
        if self.mode == 'gray' or self.mode == 'r' or self.mode == 'g' or self.mode == 'b':
            self.channels = 1
            self.database = np.zeros(shape=(h, w, self.size), dtype=np.float32)
        elif self.mode == 'rgb':
            self.channels = 3
            self.database = np.zeros(shape=(h, w, self.channels, self.size), dtype=np.float32)

    def load_from_dir(self, dir_path):
        print('Loading data...')
        (h,w) = self.img_shape
        try:
            for root, dirs, files in os.walk(dir_path):
                for name in files:
                    if self.index == self.size:
                        break
                    elif self.index < self.index_start:
                        print(self.index - self.index_start)
                        self.index += 1
                    else:
                            try:
                                if self.mode == 'gray':
                                    # Should there be an xml-file?
                                    if self.has_xml:
                                        # Generate name of coorespondig xml-file
                                        file_name, _ = name.split('.')
                                        xml_path = os.path.join(root, file_name + '.xml')
                                        # Check if the xml-file excists
                                        if not os.path.isfile(xml_path):
                                            raise Exception('\n Error: The image has no matching xml-file. \n')
                                        # Save the path to the xml-file if it excists.
                                        self.xml_files[self.true_index] = xml_path
                                    img = cv2.imread(os.path.join(root, name), cv2.IMREAD_GRAYSCALE)
                                    if img.shape[0] != self.img_shape[0] or img.shape[1] != self.img_shape[1]:
                                        img = cv2.resize(img, (w,h), interpolation=cv2.INTER_CUBIC)
                                    self.database[...,self.true_index] = img
                                elif self.mode == 'rgb':
                                    if self.has_xml:
                                        file_name, _ = name.split('.')
                                        xml_path = os.path.join(root, file_name + '.xml')
                                        if not os.path.isfile(xml_path): # Only load images with cooresponding xml-file
                                            raise Exception('\n Error: The image has no matching xml-file. \n')
                                        self.xml_files[self.true_index] = xml_path
                                    img = cv2.imread(os.path.join(root, name))
                                    if img.shape[0] != self.img_shape[0] or img.shape[1] != self.img_shape[1]:
                                        img = cv2.resize(img, (w,h), interpolation=cv2.INTER_CUBIC)
                                    self.database[...,self.true_index] = img
                                self.index += 1
                                self.true_index = self.index - self.index_start
                            except Exception as e:
                                print(e)

            print('Loading completed, last index: ', self.true_index)

        except FileNotFoundError:
            print('No directory found')

        data = np.delete(self.database, np.s_[self.index::], -1)
        self.delete_database()
        self.database = data

    def load_from_file(self, file_path):
        data = np.load(file=file_path, allow_pickle=True)
        assert data.shape[0:2] == self.img_shape, "The shape of the images in the file does not match the database shape."
        self.index = data.shape[-1]
        self.size = data.shape[-1]
        self.database = data

    def save_to_file(self, file_path):
        np.save(file=file_path, arr=self.database)

    def normalize(self, range=(0,1)):
        scale = (range[1] - range[0]) / 255
        self.database = np.multiply(self.database, scale)
        self.database = self.database - abs(range[0])

    def display_database(self):
        for i in range(self.index):
            name = 'Display: Image number' + str(i)
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.imshow(name, self.database[...,i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def manually_update_database(self, data):
        #assert data.shape[:,...] =
        if data.shape[-1] > self.size:
            self.database = data[...,0:self.size]
            self.index = self.size
        else:
            self.database = data
            self.index = data.shape[-1]

    def delete_database(self):
        self.database = None

    def merge(self, imgs):
        assert imgs.img_shape == self.img_shape, 'Cannot merge databases with images of different size!'
        assert imgs.channels == self.channels, 'Cannot merge databases with images of different depth!'
        self.database = np.concatenate((self.database, imgs.database), -1)
        self.index = self.index + imgs.index
        self.size = self.index

    def shuffle_images(self):
        np.random.shuffle(self.database.T)

def load_imgs():
    imgs_rgb = ImageLoader(scale=0.4, mode='rgb', size=3000, index_start=0)
    imgs_rgb.load_from_dir(dir_path='hurtigruta_temp')
    imgs_rgb.normalize()
    imgs_rgb.shuffle_images()
    imgs_rgb.save_to_file(file_path='Datasets_np/hurtigruta_temp_rgb_1_train.npy')

    imgs_gray = ImageLoader(scale=0.4, mode='gray', size=3000, index_start=0)
    imgs_gray.load_from_dir(dir_path='hurtigruta_temp')
    imgs_gray.normalize()
    imgs_gray.shuffle_images()
    imgs_gray.save_to_file(file_path='Datasets_np/hurtigruta_temp_gray_1_train.npy')


if __name__ == "__main__":
    load_imgs()
