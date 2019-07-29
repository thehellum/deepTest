from __future__ import print_function
import numpy as np
import os
#from epoch_model import build_cnn, build_InceptionV3
#from scipy.misc import imread, imresize
from imageio import imread
#import Image # resize
import cv2

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
# import numpy as np
# import cv2

# from PIL import Image
from PIL import ImageEnhance
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2



def read_transformed_image(image, image_size):

    img = image
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

def image_translation(img, params):

    rows, cols, ch = img.shape

    M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_scale(img, params):

    res = cv2.resize(img, None, fx=params[0], fy=params[1], interpolation=cv2.INTER_CUBIC)
    return res

def image_shear(img, params):
    rows, cols, ch = img.shape
    factor = params*(-1.0)
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_rotation(img, params):
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), params, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_contrast(img, params):
    alpha = params
    new_img = cv2.multiply(img, np.array([alpha]))                    # mul_img = img*alpha
    #new_img = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta

    return new_img

def image_brightness(img, params):
    beta = params
    new_img = cv2.add(img, beta)                                  # new_img = img*alpha + beta

    return new_img

def image_blur(img, params):
    blur = []
    if params == 1:
        blur = cv2.blur(img, (3, 3))
    if params == 2:
        blur = cv2.blur(img, (4, 4))
    if params == 3:
        blur = cv2.blur(img, (5, 5))
    if params == 4:
        blur = cv2.GaussianBlur(img, (3, 3), 0)
    if params == 5:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
    if params == 6:
        blur = cv2.GaussianBlur(img, (7, 7), 0)
    if params == 7:
        blur = cv2.medianBlur(img, 3)
    if params == 8:
        blur = cv2.medianBlur(img, 5)
    if params == 9:
        blur = cv2.blur(img, (6, 6))
    if params == 10:
        blur = cv2.bilateralFilter(img, 9, 75, 75)
    return blur

def rotation(img, params):

    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), params[0], 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_brightness1(img, params):
    w = img.shape[1]
    h = img.shape[0]
    if params > 0:
        for xi in xrange(0, w):
            for xj in xrange(0, h):
                if 255-img[xj, xi, 0] < params:
                    img[xj, xi, 0] = 255
                else:
                    img[xj, xi, 0] = img[xj, xi, 0] + params
                if 255-img[xj, xi, 1] < params:
                    img[xj, xi, 1] = 255
                else:
                    img[xj, xi, 1] = img[xj, xi, 1] + params
                if 255-img[xj, xi, 2] < params:
                    img[xj, xi, 2] = 255
                else:
                    img[xj, xi, 2] = img[xj, xi, 2] + params
    if params < 0:
        params = params*(-1)
        for xi in xrange(0, w):
            for xj in xrange(0, h):
                if img[xj, xi, 0] - 0 < params:
                    img[xj, xi, 0] = 0
                else:
                    img[xj, xi, 0] = img[xj, xi, 0] - params
                if img[xj, xi, 1] - 0 < params:
                    img[xj, xi, 1] = 0
                else:
                    img[xj, xi, 1] = img[xj, xi, 1] - params
                if img[xj, xi, 2] - 0 < params:
                    img[xj, xi, 2] = 0
                else:
                    img[xj, xi, 2] = img[xj, xi, 2] - params

    return img

def image_brightness2(img, params):
    beta = params
    b, g, r = cv2.split(img)
    b = cv2.add(b, beta)
    g = cv2.add(g, beta)
    r = cv2.add(r, beta)
    new_img = cv2.merge((b, g, r))
    return new_img


# FOG
# -------------------------------------------------------------

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




def generate_cloud_pattern(im_sizeX, im_sizeY):
    turbulence_pattern = make_fog(im_sizeX,im_sizeY)
    turb_img = Image.fromarray(np.dstack([turbulence_pattern.astype(np.uint8)]*3))
    return turb_img


# img = Image.open('axis0002.jpg')
# width, height = img.size

# cloud = generate_cloud_pattern(width,height)

# img2 = Image.blend(img, cloud, 0.6)
# img2.show()


# RAIN
# --------------------------------------------------------

def visualize(img,title):
    height, width = img.shape[:2]
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title,(int(width/2), int(height/2)) )
    cv2.moveWindow(title, 20,29)
    cv2.imshow(title, img)
    cv2.waitKey(0)

def generate_random_lines(imshape,slant,drop_length):
    drops=[]
    for i in range(500):
        ## If You want heavy rain, try increasing this                  
        x= np.random.randint(0,imshape[1]-slant)        
        y= np.random.randint(0,imshape[0]-drop_length)        
        drops.append((x,y))    
    return drops            
    

def add_rain(image):        
    imshape = image.shape    
    slant_extreme=20 
    slant= np.random.randint(-slant_extreme,slant_extreme)     
    drop_length=10    
    drop_width=3    
    drop_color=(100,100,100) ## a shade of gray    
    rain_drops= generate_random_lines(imshape,slant,drop_length)        
    overlay = image.copy()
    cv2.imshow('overlay', overlay)
    cv2.imshow('img', image)
    cv2.waitKey(0)
    for rain_drop in rain_drops:        
        cv2.line(overlay,(rain_drop[0],rain_drop[1]),
        (rain_drop[0]+slant,rain_drop[1]+drop_length),
        drop_color,drop_width)  


    cv2.imshow('overlay', overlay)
    cv2.imshow('img', image)
    cv2.waitKey(0)
    image = cv2.addWeighted(overlay, 0.7, image, 0.3,0)
    image= cv2.blur(image,(8,8)) ## rainy view are blurry        
    
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB    
    
    return image_RGB

def darken_image(img):
    height, width = img.shape[:2]
    black = np.zeros((height,width,3), np.uint8)
    img2 = cv2.addWeighted(img, 0.6, black, 0.4,0)
    
    return img2



# img = cv2.imread('axis0002.jpg')
# img = darken_image(img)
# rainy_img = add_rain(img)
# visualize(rainy_img, 'result')
