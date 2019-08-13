import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import xml.etree.ElementTree as ElementTree
from random import shuffle
from utils.augmentation.bndbox_utilities import update_bndbox, get_bndbox, draw_bndbox, iou, rotate_bounding_box
from utils.augmentation.imageLoader import ImageLoader


def generate_random_lines(imshape, slant, drop_length, number_of_drops):
    drops=[]
    for i in range(number_of_drops):
        x = np.random.randint(0,imshape[1]-slant)
        y = np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))
    return drops

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

def visualize(img):
    cv2.namedWindow('Img', cv2.WINDOW_NORMAL)
    cv2.imshow('Img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class DatasetAugmenter():
    def __init__(self, imgs, separate_folders=True, path=sys.path[0], use_original_names=True):
        self.data = imgs.database
        self.size = imgs.index
        self.global_img_count = 0
        self.img_shape = (imgs.img_shape[0], imgs.img_shape[1])
        self.img_channels = imgs.channels
        self.path = path
        self.separate_folders = separate_folders
        self.xml_file_paths = imgs.xml_files
        self.use_original_names = use_original_names

    def __call__(self):
        ## Basic augmentation
        data_gaussian = self.add_gaussian_noise(mu=0, sigma=20)
        self.save_to_dir(data=data_gaussian, dir='gaus')

        data_bb = self.add_black_box()
        self.save_to_dir(data=data_bb, dir='bb')

        intensity_shifts=[-150,-100,-50,25,75]
        for shift in intensity_shifts:
            data_brightness = self.edit_brightness(shift)
            dir_name = 'brightness_' + str(shift)
            self.save_to_dir(data=data_brightness, dir=dir_name)

        n_fog_states = 2
        fog_state = {'offset': [0,0],
                    'fog_intenisity': [1,1.2],
                     'pixel_intenisty_cutoff': [210,170],
                     'texture_variation_gain': [50,30],
                     'texture_variation_frequency': [-0.0009,0],
             'noise_variane': [0.01,0.1]}
        for state in range(n_fog_states):
            data_dense_fog = self.add_dense_fog(fog_state=fog_state, state=state)
            dir_name = 'd_fog_' + str(state)
            self.save_to_dir(data=data_dense_fog, dir=dir_name)


        data_sparse_fog = self.add_sparse_fog(fog_intensity=0.6)
        self.save_to_dir(data=data_sparse_fog, dir='s_fog')

        data_gaussian_blur = self.gaussian_blur(ksize=(5,5), sigmaX=7, sigmaY=7)
        self.save_to_dir(data=data_gaussian_blur, dir='blur')

        thetas=[-10,10]
        for theta in thetas:
            data_rotate, M = self.rotate(theta=theta)
            dir_name = 'rotate_' + str(theta)
            self.save_to_dir(data=data_rotate, dir=dir_name, M=M)

        data_rain = self.add_sparse_fog(fog_intensity=0.6)
        data_rain = self.add_rain(number_of_drops=1500, drop_length=4, drop_width=1, data=data_rain)
        data_rain = self.gaussian_blur(ksize=(5,5), sigmaX=9, sigmaY=9, data=data_rain)
        self.save_to_dir(data=data_rain, dir='rain')

        data_snow = self.add_sparse_fog(fog_intensity=0.6)
        data_snow = self.add_snow(number_of_drops=1000, drop_length=2, drop_width=2, data=data_snow)
        data_snow = self.gaussian_blur(data=data_snow)
        self.save_to_dir(data=data_snow, dir='snow')

        #data_rain_on_lens = self.add_rain_on_lens()
        #self.save_to_dir(data=data_rain_on_lens, dir='rain_on_lens')


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

    def orignial_file_name(self, index):
        path = self.xml_file_paths[index]
        name =os.path.basename(path)
        #name = path.split('\\')[-1]
        return name[:-4]

    def save_image(self, img, dir, index, M=np.array([None])):
        tree = self.load_xml_file(index=index)
        if M.all() != None:
            rotate_bounding_box(tree=tree, M=M)

        if self.separate_folders:
            if self.use_original_names:
                name = self.orignial_file_name(index)
            else:
                name = self.generate_name(seed=index, file_type='')

            file_name_img = os.path.join(self.path, dir, name + '.jpg')
            file_name_xml = os.path.join(self.path, dir, name + '.xml')

            cv2.imwrite(file_name_img, img)
            tree.write(file_name_xml)
        else:
            file_name_img_global = self.generate_name(seed=self.global_img_count, file_type='.jpg')
            file_name_xml_global = self.generate_name(seed=self.global_img_count, file_type='.xml')

            cv2.imwrite(os.path.join(self.path, file_name_img_global), img)
            tree.write(os.path.join(self.path, file_name_xml_global))

            self.global_img_count += 1

    def save_to_dir(self, data, dir, M=None):
        if self.separate_folders:
            self.create_directory(name=dir)
        for i in range(data.shape[-1]):
            img = data[...,i]
            if M is not None:
                self.save_image(img=img, dir=dir, index=i, M=M)
            else:
                self.save_image(img=img, dir=dir, index=i)

    def non_augmented(self, data=None):
        if data is None:
            data = np.copy(a=self.data)
        assert len(data.shape) == 4 or len(data.shape) == 3
        assert data.shape[0] != 0
        return data

    def add_gaussian_noise(self, mu, sigma, data=None):
        if data is None:
            data = np.copy(a=self.data)
        for i in range(self.size):
            gaussian_noise = np.random.normal(mu, sigma, (self.img_shape[0], self.img_shape[1], self.img_channels))
            img = data[...,i]
            img += gaussian_noise
        return data

    def add_black_box(self, data=None):
        if data is None:
            data = np.copy(a=self.data)

        h,w = self.img_shape
        for k in range(self.size):
            tree = self.load_xml_file(index=k)
            bndbox, _ = get_bndbox(tree, None)
            img = data[...,k]
            box_approved = True
            counter = 0
            while box_approved:
                x = int(np.random.uniform(low=50, high=150, size=1))
                y = int(np.random.uniform(low=50, high=150, size=1))
                i = int(np.random.uniform(low=x//2, high=(h-(x//2)), size=1))
                j = int(np.random.uniform(low=y//2, high=(w-(y//2)), size=1))
                xmin,ymin,xmax,ymax = j-y//2, i-x//2, j+y//2, i+x//2
                for (_xmin,_ymin,_xmax,_ymax) in bndbox:
                    IoU = iou([_xmin,_ymin,_xmax,_ymax], [xmin,ymin,xmax,ymax])
                    if IoU < 0.01:
                        img[i-x//2:i+x//2, j-y//2:j+y//2, :] = 0
                        box_approved = False
                    elif counter > 10:
                        box_approved = False
                    else:
                        pass
                    counter += 1
        return data

    def edit_brightness(self, shift, data=None):
        if data is None:
            data = np.copy(a=self.data)
        for j in range(self.size):
            noise = float(np.random.uniform(low=5, high=5))
            img = data[...,j]
            img += (shift + noise)
            img[img < 0] = 0
            img[img > 255] = 255
        return data

    def add_dense_fog(self, fog_state, state, remove_bright=True, data=None):
        # Name all the hyperparameters and make a procedure for generating suitable hyperparameters randomly
        if data is None:
            data = np.copy(a=self.data)

        if self.img_channels == 3:
            shape = (self.img_shape[0], self.img_shape[1], self.img_channels)
        else:
            shape = (self.img_shape[0], self.img_shape[1])

        new_data = np.zeros((shape[0],shape[1],shape[2],0), dtype=np.float32)

        offset = fog_state['offset'][state]
        fog_intenisity = fog_state['fog_intenisity'][state]
        pixel_intenisty_cutoff = fog_state['pixel_intenisty_cutoff'][state]
        texture_variation_gain = fog_state['texture_variation_gain'][state]
        texture_variation_frequency = fog_state['texture_variation_frequency'][state]
        noise_variane = fog_state['noise_variane'][state]

        fog_filter = np.zeros(shape)

        for i in range(1,shape[0]+1):
            if i >= offset:
                #fog_filter[-i,:,:] = (i-(offset-1))*(i)*np.ones((shape[1],shape[2]))
                fog_filter[-i,:,:] = ((i-offset) + 10)*np.ones((shape[1],shape[2]))
            elif i < offset:
                #fog_filter[-i,:,:] = offset*i*np.ones((shape[1],shape[2]))
                fog_filter[-i,:,:] = 5*np.ones((shape[1],shape[2]))

        fog_filter /= (fog_filter.max()/fog_intenisity)
        fog_filter *= 255

        for j in range(self.size):
            img = data[...,j]

            h,w,_ = img.shape
            intenisty = np.average(a=img[h//2:-1,:,:], axis=(0,1,2))
            intensity_limit = 120

            if intenisty > intensity_limit:
                pass
            else:
                shift = np.average(a=img, axis=(0,1,2)) - 150 + int(np.random.uniform(low=-5,high=5))
                img -= int(shift)
                img[img < 0] = 0

                #img += fog_filter
                img = cv2.addWeighted(img, 0.3, fog_filter.astype(np.float32), 0.7,0)
                img[img > pixel_intenisty_cutoff] = pixel_intenisty_cutoff
                x = np.arange(shape[1])
                y = texture_variation_gain*np.sin(2*np.pi*x*texture_variation_frequency)
                for k in range(1,shape[0]+1):
                    f = (k/shape[0])*texture_variation_frequency
                    var = (k/shape[0])*noise_variane
                    y = texture_variation_gain*np.sin(2*np.pi*x*f) + np.random.normal(loc=0, scale=var, size=shape[1])
                    img[-k,:,:] += np.concatenate((y.reshape(shape[1],1),y.reshape(shape[1],1),y.reshape(shape[1],1)),1)
                data[...,j] = img
                new_data = np.concatenate((new_data, np.expand_dims(img, axis=-1)), axis=-1)
        if remove_bright:
            return new_data
        else:
            return data

    def add_sparse_fog(self, fog_intensity, data=None):
        if data is None:
            data = np.copy(a=self.data)

        for i in range(self.size):
            img = data[...,i]
            h,w  = self.img_shape
            cloud_pattern = make_fog(h,w)
            cloud = np.dstack([cloud_pattern.T.astype(np.float32)]*3)
            data[...,i] = cv2.addWeighted(img, fog_intensity, cloud, 0.4, 0)
        return data

    def gaussian_blur(self, ksize=(5,5), sigmaX=7, sigmaY=7, data=None):
        if data is None:
            data = np.copy(a=self.data)
        for i in range(self.size):
            img = data[...,i]
            data[...,i] = cv2.GaussianBlur(src=img, ksize=ksize, sigmaX=sigmaX, sigmaY=sigmaY)
        return data

    def rotate(self, theta, data=None):
        if data is None:
            data = np.copy(a=self.data)
        h,w = self.img_shape
        noise = float(np.random.uniform(low=5, high=5))
        theta += noise
        M = cv2.getRotationMatrix2D(center=(w//2,h//2), angle=theta, scale=1)
        for i in range(self.size):
            tree = self.load_xml_file(index=i)
            img = data[...,i]
            data[...,i] = cv2.warpAffine(img, M, (w,h))
        return data, M

    def add_rain(self, number_of_drops=1500, drop_length=4, drop_width=1, data=None):
        if data is None:
            data = np.copy(a=self.data)

        slant_extreme = 1
        slant = np.random.randint(-slant_extreme,slant_extreme)
        drop_length = drop_length
        drop_width = drop_width
        number_of_drops = number_of_drops
        drop_color = (100,100,100) ## a shade of gray
        for i in range(self.size):
            rain_drops = generate_random_lines(self.img_shape, slant, drop_length, number_of_drops)
            img = data[...,i]
            overlay = img.copy()
            for rain_drop in rain_drops:
                cv2.line(overlay,(rain_drop[0],rain_drop[1]),
                (rain_drop[0]+slant,rain_drop[1]+drop_length),
                drop_color,drop_width)
            img = cv2.addWeighted(overlay, 0.7, img, 0.3,0)
            #img = cv2.blur(img,(3,3)) ## rainy view are blurry
            img_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) ## Conversion to HLS
            data[...,i] = cv2.cvtColor(img_HLS, cv2.COLOR_HLS2RGB) ## Conversion to RGB
        return data

    def add_snow(self, number_of_drops=1000, drop_length=1, drop_width=1, data=None):
        if data is None:
            data = np.copy(a=self.data)
        slant_extreme=1
        slant= np.random.randint(-slant_extreme,slant_extreme)
        drop_length=drop_length
        drop_width=drop_width
        number_of_drops = number_of_drops
        drop_color=(255,255,255) ## a shade of gray
        for i in range(self.size):
            rain_drops = generate_random_lines(self.img_shape, slant, drop_length, number_of_drops)
            img = data[...,i]
            overlay = img.copy()
            for rain_drop in rain_drops:
                cv2.line(overlay,(rain_drop[0],rain_drop[1]),
                (rain_drop[0]+slant,rain_drop[1]+drop_length),
                drop_color,drop_width)
            img = cv2.addWeighted(overlay, 0.7, img, 0.3,0)
            img = cv2.blur(img,(3,3)) ## rainy view are blurry
            img_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) ## Conversion to HLS
            data[...,i] = cv2.cvtColor(img_HLS, cv2.COLOR_HLS2RGB) ## Conversion to RGB
        return data

    # Not done yet
    def add_rain_on_lens(self, drops=30, data=None):
        if data is None:
            data = np.copy(a=self.data)
        h,w = self.img_shape
        for i in range(self.size):
            print('Augmenting image', i)
            img = data[...,i]
            mask = np.zeros((h,w,3), dtype=np.float32)
            centers = np.random.uniform(low=10, high=w-10, size=(drops,2))
            for c0,c1 in centers:
                diameter = np.random.uniform(low=10, high=50)
                for j in range(h):
                    for k in range(w):
                        if (j-c0)**2 + (k-c1)**2 < diameter**2:
                            mask[j,k,:] = img[j,k,:]
                            img[j,k,:] = 0
            mask = cv2.GaussianBlur(src=mask, ksize=(7,7), sigmaX=15, sigmaY=15)
            merged = cv2.addWeighted(src1=img, alpha=0.5, src2=mask, beta=0.5, gamma=-0.2)
            data[...,i] = cv2.GaussianBlur(src=merged, ksize=(3,3), sigmaX=3, sigmaY=3)
        return data

    def visualize_bndboxes(self, n=2):
        dirs = os.listdir(self.path)
        shuffle(dirs)
        for dir in dirs:
            path = os.path.join(self.path, dir)
            for root, _, file_list in os.walk(path):
                shuffle(file_list)
                count = 0
                for file in file_list:
                    if '.jpg' in file:
                        if count == n:
                            break
                        path = os.path.join(root, file)
                        path_xml = os.path.join(root, file[:-4] + '.xml')
                        tree = ElementTree.parse(path_xml)
                        img = cv2.imread(path)
                        (h,w,_) = img.shape
                        bndbox = get_bndbox(tree, None)
                        for box in bndbox:
                            xmin, ymin, xmax, ymax = box
                            draw_bndbox(img, xmin, ymin, xmax, ymax)
                        visualize(img)
                        count += 1


def augment(data_path, save_path):
    num_xml = len([name for name in os.listdir(data_path) if '.xml' in name])
    imgs_rgb = ImageLoader(shape=(720,1280), scale=1, mode='rgb', size=num_xml)
    imgs_rgb.load_from_dir(dir_path=data_path)

    # Don't normalize!
    augmenter = DatasetAugmenter(imgs=imgs_rgb, separate_folders=True, path=save_path, use_original_names=True)
    augmenter()
    # augmenter.visualize_bndboxes()


# if __name__ == '__main__':
#     path_save_to = os.path.join(sys.path[0], 'Data_augmentation')
#     path_data = os.path.join(sys.path[0], 'Dataset_full')

#     imgs_rgb = ImageLoader(shape=(720,1280), scale=1, mode='rgb', size=10)
#     imgs_rgb.load_from_dir(dir_path=data_path)

#     # Don't normalize!
#     augmenter = DatasetAugmenter(imgs=imgs_rgb, separate_folders=True, path=save_path, use_original_names=True)
#     augmenter()
#     augmenter.visualize_bndboxes()
