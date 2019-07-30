from PIL import Image
#from PIL import ImageEnhance
import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import xml.etree.ElementTree as ElementTree
from utils.dataframe import XmlDictConfig
import pandas as pd
from utils.dataset_feature_extraction import ImageLoader, visualize

class DatasetAugmenter():
    def __init__(self, imgs, separate_folders=True, path=sys.path[0]):
        self.database = imgs.database
        self.size = imgs.index
        self.img_count = 0
        self.img_shape = (imgs.img_shape[0], imgs.img_shape[1])
        self.img_channels = imgs.channels
        self.path = path
        self.separate_folders = separate_folders
        self.xml_file_paths = imgs.xml_files

    def generate_name(self, seed, file_type='.jpg'):
        if len(str(seed)) == 1:
            return '0000' + str(seed) + file_type
        elif len(str(seed)) == 2:
            return '000' + str(seed) + file_type
        elif len(str(seed)) == 3:
            return '00' + str(seed) + file_type
        elif len(str(seed)) == 4:
            return '0' + str(seed) + file_type
        else:
            return str(seed) + file_type

    def create_directory(self, name):
        path = os.path.join(self.path, name)
        if not os.path.exists(path):
            os.makedirs(path)

    def load_xml_file(self, index):
        path = self.xml_file_paths[index]
        tree = ElementTree.parse(path)
        return tree

    def original_file_name(self, index):
        path = self.xml_file_paths[index]
        return os.path.basename(path)[:-4]

    def get_bndbox(self, tree):
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

    def update_bndbox(self, tree, bndbox):
        root = tree.getroot()
        object_count = len(bndbox)
        for i in range(object_count):
            xmin, ymin, xmax, ymax = bndbox[i]
            root[6+i][4][0].text = str(xmin)
            root[6+i][4][1].text = str(ymin)
            root[6+i][4][2].text = str(xmax)
            root[6+i][4][3].text = str(ymax)

    def save_image(self, img, dir, name, index, M=np.array([None])):
        tree = self.load_xml_file(index=index)
        if M.all() != None:
            bndbox = self.get_bndbox(tree)
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

            self.update_bndbox(tree, new_bndbox)

        if self.separate_folders:
            o_name = self.original_file_name(index)
            cv2.imwrite(os.path.join(self.path, dir, o_name + '.jpg'), img)
            tree.write(os.path.join(self.path, dir, o_name + '.xml'))
        else:
            global_name = self.generate_name(seed=self.img_count)
            global_name_xml = self.generate_name(seed=self.img_count, file_type='.xml')
            cv2.imwrite(os.path.join(self.path, global_name), img)
            tree.write(os.path.join(self.path, global_name_xml))
            self.img_count += 1

    def add_non_augmented(self):
        if self.separate_folders:
            self.create_directory(name='original')

        for i in range(self.size):
            name = self.generate_name(seed=i)

            img = np.copy(a=self.database[...,i])
            self.save_image(img=img, dir='original', name=name, index=i)

    def add_gaussian_noise(self, mu, sigma):
        if self.separate_folders:
            self.create_directory(name='gaussian_noise')

        for i in range(self.size):
            name = self.generate_name(seed=i)
            gaussian_noise = np.random.normal(mu, sigma, (self.img_shape[0], self.img_shape[1], self.img_channels))
            img = np.copy(a=self.database[...,i])
            img += gaussian_noise
            self.save_image(img=img, dir='gaussian_noise', name=name, index=i)

    def add_black_box(self):
        if self.separate_folders:
            self.create_directory(name='black_box')
        h,w = self.img_shape
        for k in range(self.size):
            tree = self.load_xml_file(index=k)
            bndbox = self.get_bndbox(tree)
            name = self.generate_name(seed=k)
            img = np.copy(a=self.database[...,k])
            box_approved = True
            counter = 0
            while box_approved:
                x = int(np.random.uniform(low=50, high=150, size=1))
                y = int(np.random.uniform(low=50, high=150, size=1))
                i = int(np.random.uniform(low=x//2, high=(h-(x//2)), size=1))
                j = int(np.random.uniform(low=y//2, high=(w-(y//2)), size=1))
                xmin,ymin,xmax,ymax = j-y//2, i-x//2, j+y//2, i+x//2
                for box in bndbox:
                    IoU = iou(box, [xmin,ymin,xmax,ymax])
                    if IoU < 0.01:
                        img[i-x//2:i+x//2, j-y//2:j+y//2, :] = 0
                        box_approved = False
                    elif counter > 10:
                        box_approved = False
                    else:
                        pass
                    counter += 1
            self.save_image(img=img, dir='black_box', name=name, index=k)

    def edit_brightness(self, intensity_shifts):
        for shift in intensity_shifts:
            dir_name = 'brightness' + '_' + str(shift)
            if self.separate_folders:
                self.create_directory(name=dir_name)
            for j in range(self.size):
                file_name = self.generate_name(seed=+j)
                img = np.copy(self.database[...,j])
                img += shift
                img[img < 0] = 0
                img[img > 255] = 255
                self.save_image(img=img, dir=dir_name, name=file_name, index=j)

    def add_dense_fog(self, fog_state, n_states):
        # Name all the hyperparameters and make a procedure for generating suitable hyperparameters randomly
        if self.separate_folders:
            self.create_directory(name='d_fog')

        if self.img_channels == 3:
            shape = (self.img_shape[0], self.img_shape[1], self.img_channels)
        else:
            shape = (self.img_shape[0], self.img_shape[1])

        for state in range(n_states):
            offset = fog_state['offset'][state]
            fog_intenisity = fog_state['fog_intenisity'][state]
            pixel_intenisty_cutoff = fog_state['pixel_intenisty_cutoff'][state]
            texture_variation_gain = fog_state['texture_variation_gain'][state]
            texture_variation_frequency = fog_state['texture_variation_frequency'][state]
            noise_variane = fog_state['noise_variane'][state]

            fog_filter = np.zeros(shape)

            for i in range(1,shape[0]+1):
                if i >= offset:
                    fog_filter[-i,:,:] = (i-(offset-1))*(i)*np.ones((shape[1],shape[2]))
                elif i < offset:
                    fog_filter[-i,:,:] = offset*i*np.ones((shape[1],shape[2]))

            fog_filter /= (fog_filter.max()/fog_intenisity)
            fog_filter *= 255

            for j in range(self.size):
                name = self.generate_name(seed=state*self.size+j)
                img = np.copy(self.database[...,j])
                if img.max() > 200:
                    img -= img.max()*0.3
                    img[img < 0] = 0
                foggy = img + fog_filter
                foggy[foggy > pixel_intenisty_cutoff] = pixel_intenisty_cutoff

                x = np.arange(shape[1])
                y = texture_variation_gain*np.sin(2*np.pi*x*texture_variation_frequency)
                for k in range(1,shape[0]+1):
                    f = (k/shape[0])*texture_variation_frequency
                    var = (k/shape[0])*noise_variane
                    y = texture_variation_gain*np.sin(2*np.pi*x*f) + np.random.normal(loc=0, scale=var, size=shape[1])
                    foggy[-k,:,:] += np.concatenate((y.reshape(shape[1],1),y.reshape(shape[1],1),y.reshape(shape[1],1)),1)
                self.save_image(img=foggy, dir='d_fog', name=name, index=j)

    def add_sparse_fog(self, fog_intensity):
        if self.separate_folders:
            self.create_directory(name='s_fog')

        for i in range(self.size):
            name = self.generate_name(seed=i)
            img = np.copy(a=self.database[...,i])
            h,w  = self.img_shape
            cloud = generate_cloud_pattern(h,w)
            img2 = cv2.addWeighted(img, fog_intensity, cloud, 0.4, 0)
            self.save_image(img=img2, dir='s_fog', name=name, index=i)

    def gaussian_blur(self, ksize=(5,5), sigmaX=5, sigmaY=5):
        if self.separate_folders:
            self.create_directory(name='blur')
        for i in range(self.size):
            name = self.generate_name(seed=i)
            img = np.copy(a=self.database[...,i])
            img_blurred = cv2.GaussianBlur(src=img, ksize=ksize, sigmaX=sigmaX, sigmaY=sigmaY)
            self.save_image(img=img_blurred, dir='blur', name=name, index=i)

    def rotate(self, theta):
        if self.separate_folders:
            self.create_directory(name='rotate')
        h,w = self.img_shape
        channels = self.img_channels
        M = cv2.getRotationMatrix2D(center=(h//2,w//2), angle=theta, scale=1)
        for i in range(self.size):
            name = self.generate_name(seed=i)
            img = np.copy(a=self.database[...,i])
            img_rotated = cv2.warpAffine(img, M, (w,h))
            self.save_image(img=img_rotated, dir='rotate', name=name, index=i, M=M)

    def add_rain(self):
        if self.separate_folders:
            self.create_directory(name='rain')
        slant_extreme=1
        slant= np.random.randint(-slant_extreme,slant_extreme)
        drop_length=1
        drop_width=1
        number_of_drops = 1000
        drop_color=(100,100,100) ## a shade of gray
        for i in range(self.size):
            name = self.generate_name(seed=i)
            rain_drops = generate_random_lines(self.img_shape, slant, drop_length, number_of_drops)
            img = np.copy(a=self.database[...,i])
            overlay = img.copy()
            for rain_drop in rain_drops:
                cv2.line(overlay,(rain_drop[0],rain_drop[1]),
                (rain_drop[0]+slant,rain_drop[1]+drop_length),
                drop_color,drop_width)
            img = cv2.addWeighted(overlay, 0.7, img, 0.3,0)
            img = cv2.blur(img,(3,3)) ## rainy view are blurry
            img_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) ## Conversion to HLS
            img_RGB = cv2.cvtColor(img_HLS, cv2.COLOR_HLS2RGB) ## Conversion to RGB
            self.save_image(img=img_RGB, dir='rain', name=name, index=i)

    def add_snow(self):
        if self.separate_folders:
            self.create_directory(name='snow')
        slant_extreme=1
        slant= np.random.randint(-slant_extreme,slant_extreme)
        drop_length=2
        drop_width=2
        number_of_drops = 1000
        drop_color=(255,255,255) ## a shade of gray
        for i in range(self.size):
            name = self.generate_name(seed=i)
            rain_drops = generate_random_lines(self.img_shape, slant, drop_length, number_of_drops)
            img = np.copy(a=self.database[...,i])
            overlay = img.copy()
            for rain_drop in rain_drops:
                cv2.line(overlay,(rain_drop[0],rain_drop[1]),
                (rain_drop[0]+slant,rain_drop[1]+drop_length),
                drop_color,drop_width)
            img = cv2.addWeighted(overlay, 0.7, img, 0.3,0)
            img = cv2.blur(img,(3,3)) ## rainy view are blurry
            img_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) ## Conversion to HLS
            img_RGB = cv2.cvtColor(img_HLS, cv2.COLOR_HLS2RGB) ## Conversion to RGB
            self.save_image(img=img_RGB, dir='snow', name=name, index=i)

    # Not finished yet
    def add_rain_on_lens(self):
        path = os.path.join(self.path, 'rain_lense')
        if not os.path.exists(path):
            os.makedirs(path)
        h,w = self.img_shape
        channels = self.img_channels
        for i in range(self.size):
            name = self.generate_name(seed=i)
            img = np.copy(a=self.database[...,i])
            img = img - 0.2
            mask = np.zeros((h,w,3), dtype=np.float32)
            centers = np.random.uniform(low=0, high=w, size=(20,2))
            for c0,c1 in centers:
                for i in range(h):
                    for j in range(w):
                        if (i-c0)**2 + (j-c1)**2 < 10**2:
                            mask[i,j,:] = img[i,j,:]
                            img[i,j,:] = 0
            #mask = mask - 0.1
            #mask[mask < 0] = 0
            mask = cv2.GaussianBlur(src=mask, ksize=(7,7), sigmaX=15, sigmaY=15)
            #visualize(mask/255)
            merged = cv2.addWeighted(src1=img, alpha=0.5, src2=mask, beta=0.5, gamma=0)
            #merged = img+mask
            merged = cv2.GaussianBlur(src=merged, ksize=(3,3), sigmaX=3, sigmaY=3)
            #visualize(merged/255)
            if self.separate_folders:
                cv2.imwrite(os.path.join(path, name), merged)

    def train_val_split2(self, split=0.8):
        val_dir = os.path.join(self.path, 'val')
        train_dir = os.path.join(self.path, 'train')
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        for root, dirs, files in os.walk(self.path):
            for name in files:
                if '.jpeg' in name or '.jpg' in name:
                    name, type = name.split('.')
                file_path = os.path.join(root, name)
                if float(np.random.uniform(low=0, high=1, size=1)) > split:
                    os.rename(file_path, os.path.join(val_dir, name))
                else:
                    os.rename(file_path, os.path.join(train_dir, name))


def generate_random_lines(imshape, slant, drop_length, number_of_drops):
    drops=[]
    for i in range(number_of_drops):
        ## If You want heavy rain, try increasing this
        x = np.random.randint(0,imshape[1]-slant)
        y = np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))
    return drops

def darken_image(img):
    height, width = img.shape[:2]
    img = img.astype(np.uint8)
    black = np.zeros((height,width,3), np.uint8)
    img2 = cv2.addWeighted(img, 0.6, black, 0.4,0)
    return img2

def generate_cloud_pattern(im_sizeX, im_sizeY):
    turbulence_pattern = make_fog(im_sizeX,im_sizeY)
    turb_img = np.dstack([turbulence_pattern.T.astype(np.float32)]*3)
    return turb_img

def make_fog(im_sizeX, im_sizeY):
    base_pattern = np.random.uniform(50,255,(im_sizeY//2, im_sizeX//2)) #base of white noise pattern
    turbulence_pattern = np.zeros((im_sizeY,im_sizeX))
    power_range = range(2, int(np.log2((im_sizeX + im_sizeY)/2)))
    for i in power_range:
        subimg_size = 2**i
        quadrant = base_pattern[:subimg_size, :subimg_size]
        upsampled_pattern = cv2.resize(quadrant, dsize=(im_sizeX,im_sizeY), interpolation= cv2.INTER_CUBIC)
        turbulence_pattern += upsampled_pattern / subimg_size
    turbulence_pattern /= sum([1 / 2**i for i in power_range])
    return turbulence_pattern

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

def visualize(img):
    cv2.namedWindow('Img', cv2.WINDOW_NORMAL)
    cv2.imshow('Img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def augment(data_path, save_path):
    imgs_rgb = ImageLoader(shape=(720,1280), scale=1, mode='rgb', size=20)
    imgs_rgb.load_from_dir(dir_path=data_path)
    # Don't normalize!
    da = DatasetAugmenter(imgs=imgs_rgb, separate_folders=True, path=save_path)

    # da.add_non_augmented()
    da.add_gaussian_noise(mu=0, sigma=20)
    da.add_black_box()
    da.edit_brightness(intensity_shifts=[-150,-100,-50,25,75])
    fog_state = {'offset': [10,0],
                'fog_intenisity': [3.0,2.0],
                 'pixel_intenisty_cutoff': [210,210],
                 'texture_variation_gain': [50,30],
                 'texture_variation_frequency': [-0.0009,0.001],
                 'noise_variane': [0.01,0.01]}
    da.add_dense_fog(fog_state, n_states=2)
    da.add_sparse_fog(fog_intensity=0.6)
    da.gaussian_blur()
    da.rotate(theta = 10)
    da.add_rain()
    da.add_snow()
    #da.add_rain_on_lens()

    #for i in range(10):
        #rain_on_lens(imgs_rgb.database[...,i])


if __name__ == '__main__':
    imgs_rgb = ImageLoader(shape=(720,1280), scale=1, mode='rgb', size=20)
    imgs_rgb.load_from_dir(dir_path='Dataset_full')
    # Don't normalize!
    da = DatasetAugmenter(imgs=imgs_rgb, separate_folders=True, path=sys.path[0] + '\Data_augmentation' )
    #train_val_split(split=0.8, path_from=os.path.join(sys.path[0], 'data', 'dnv_dataset.csv'), path_to=os.path.join(sys.path[0], 'data'))

    da.add_non_augmented()
    da.add_gaussian_noise(mu=0, sigma=20)
    da.add_black_box()
    da.edit_brightness(intensity_shifts=[-150,-100,-50,25,75])
    fog_state = {'offset': [10,0],
                'fog_intenisity': [3.0,2.0],
                 'pixel_intenisty_cutoff': [210,210],
                 'texture_variation_gain': [50,30],
                 'texture_variation_frequency': [-0.0009,0.001],
                 'noise_variane': [0.01,0.01]}
    da.add_dense_fog(fog_state, n_states=2)
    da.add_sparse_fog(fog_intensity=0.6)
    da.gaussian_blur()
    da.rotate(theta = 10)
    da.add_rain()
    da.add_snow()
    #da.add_rain_on_lens()

    #for i in range(10):
        #rain_on_lens(imgs_rgb.database[...,i])
