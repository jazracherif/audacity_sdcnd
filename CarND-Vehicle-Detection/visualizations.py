import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.externals import joblib
from feature_extraction_utilities import extract_features
from feature_extraction_utilities import get_hog_features

import pickle
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

"""
    This file implements the logic for learning a classifier based on the following features:
     - HOG Feature
     - Spatial Binning
     - Color Histogram
"""

## Load Images and divide them up into cars and notcars
CARS_FILES = ['../data/vehicles/GTI_Far/*.png',
              '../data/vehicles/GTI_Left/*.png',
              '../data/vehicles/GTI_MiddleClose/*.png',
              '../data/vehicles/GTI_Right/*.png',
              '../data/vehicles/KITTI_extracted/*.png']

NOT_CARS_FILES = [ '../data/non-vehicles/GTI/*.png',
                  '../data/non-vehicles/Extras/*.png']

cars = []
notcars = []

for f in CARS_FILES:
    cars += glob.glob(f)
for f in NOT_CARS_FILES:
    notcars += glob.glob(f)

print ("#cars:", len(cars), "#notcars:", len(notcars))
image = cv2.imread(cars[0])
print ("Image shape=",image.shape)

car_img = cv2.imread(cars[1000])
car_img = cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB)

notcar_img = cv2.imread(notcars[1])
notcar_img = cv2.cvtColor(notcar_img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(car_img)
plt.title('Car Image')
plt.subplot(122)
plt.imshow(notcar_img)
plt.title('Not Car Image')
fig.tight_layout()
plt.savefig('../output_images/car_not_car.png')



orient = 9
pix_per_cell = 8
cell_per_block = 2

################
## CAR IMAGE: DRAW RGB HOG FEATURES
car_img_rgb_r = car_img[:,:,0]
car_img_rgb_g = car_img[:,:,1]
car_img_rgb_b = car_img[:,:,2]

features, hog_image_r = get_hog_features(car_img_rgb_r, orient, pix_per_cell, cell_per_block,
                                         vis=True, feature_vec=True)
features, hog_image_g = get_hog_features(car_img_rgb_g, orient, pix_per_cell, cell_per_block,
                                         vis=True, feature_vec=True)
features, hog_image_b = get_hog_features(car_img_rgb_b, orient, pix_per_cell, cell_per_block,
                                         vis=True, feature_vec=True)

fig = plt.figure()
plt.rc('font', size=16)
fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(20, 30))
axes[0, 0].imshow(car_img)
axes[0, 0].set_title('Car in RGB')
axes[0, 1].imshow(hog_image_r)
axes[0, 1].set_title('Hog channel R')
axes[0, 2].imshow(hog_image_g)
axes[0, 2].set_title('Hog channel G')
axes[0, 3].imshow(hog_image_b)
axes[0, 3].set_title('Hog channel B')

#fig.tight_layout()
#plt.savefig('../output_images/HOG_RGB.png')

## CAR IMAGE: DRAW HLS HOG FEATURES
car_img_hls = cv2.cvtColor(car_img, cv2.COLOR_RGB2HLS)
car_img_hls_h = car_img_hls[:,:,0]
car_img_hls_l = car_img_hls[:,:,1]
car_img_hls_s = car_img_hls[:,:,2]

features, hog_image_h = get_hog_features(car_img_hls_h, orient, pix_per_cell, cell_per_block,
                     vis=True, feature_vec=True)
features, hog_image_l = get_hog_features(car_img_hls_l, orient, pix_per_cell, cell_per_block,
                                         vis=True, feature_vec=True)
features, hog_image_s = get_hog_features(car_img_hls_s, orient, pix_per_cell, cell_per_block,
                                         vis=True, feature_vec=True)

axes[1, 0].imshow(car_img_hls)
axes[1, 0].set_title('Car in HLS')
axes[1, 1].imshow(hog_image_h)
axes[1, 1].set_title('Hog channel H')
axes[1, 2].imshow(hog_image_l)
axes[1, 2].set_title('Hog channel L')
axes[1, 3].imshow(hog_image_s)
axes[1, 3].set_title('Hog channel S')

#fig.tight_layout()
#plt.savefig('../output_images/HOG_HLS.png')

## CAR IMAGE: DRAW YCrCb HOG FEATURES
car_img_YCrCb = cv2.cvtColor(car_img, cv2.COLOR_RGB2YCrCb)
car_img_YCrCb_Y = car_img_YCrCb[:,:,0]
car_img_YCrCb_Cr = car_img_YCrCb[:,:,1]
car_img_YCrCb_Cb = car_img_YCrCb[:,:,2]

features, hog_image_Y = get_hog_features(car_img_YCrCb_Y, orient, pix_per_cell, cell_per_block,
                                         vis=True, feature_vec=True)
features, hog_image_Cr = get_hog_features(car_img_YCrCb_Cr, orient, pix_per_cell, cell_per_block,
                                          vis=True, feature_vec=True)
features, hog_image_Cb = get_hog_features(car_img_YCrCb_Cb, orient, pix_per_cell, cell_per_block,
                                          vis=True, feature_vec=True)

axes[2, 0].imshow(car_img_YCrCb)
axes[2, 0].set_title('Car in YCrCb')
axes[2, 1].imshow(hog_image_Y)
axes[2, 1].set_title('Hog ch. Y')
axes[2, 2].imshow(hog_image_Cr)
axes[2, 2].set_title('Hog ch. Cr')
axes[2, 3].imshow(hog_image_Cb)
axes[2, 3].set_title('Hog ch. Cb')


########################################
## NOTCAR IMAGE: DRAW RGB HOG FEATURES

car_img_rgb_r = notcar_img[:,:,0]
car_img_rgb_g = notcar_img[:,:,1]
car_img_rgb_b = notcar_img[:,:,2]

features, hog_image_r = get_hog_features(car_img_rgb_r, orient, pix_per_cell, cell_per_block,
                                         vis=True, feature_vec=True)
features, hog_image_g = get_hog_features(car_img_rgb_g, orient, pix_per_cell, cell_per_block,
                                         vis=True, feature_vec=True)
features, hog_image_b = get_hog_features(car_img_rgb_b, orient, pix_per_cell, cell_per_block,
                                         vis=True, feature_vec=True)

axes[3, 0].imshow(notcar_img)
axes[3, 0].set_title('NotCar in RGB')
axes[3, 1].imshow(hog_image_r)
axes[3, 1].set_title('Hog ch. R')
axes[3, 2].imshow(hog_image_g)
axes[3, 2].set_title('Hog ch. G')
axes[3, 3].imshow(hog_image_b)
axes[3, 3].set_title('Hog ch. B')


## CAR IMAGE: DRAW HLS HOG FEATURES
notcar_img_hls = cv2.cvtColor(notcar_img, cv2.COLOR_RGB2HLS)
notcar_img_hls_h = notcar_img_hls[:,:,0]
notcar_img_hls_l = notcar_img_hls[:,:,1]
notcar_img_hls_s = notcar_img_hls[:,:,2]

features, hog_image_h = get_hog_features(notcar_img_hls_h, orient, pix_per_cell, cell_per_block,
                                         vis=True, feature_vec=True)
features, hog_image_l = get_hog_features(notcar_img_hls_l, orient, pix_per_cell, cell_per_block,
                                         vis=True, feature_vec=True)
features, hog_image_s = get_hog_features(notcar_img_hls_s, orient, pix_per_cell, cell_per_block,
                                         vis=True, feature_vec=True)

axes[4, 0].imshow(notcar_img_hls)
axes[4, 0].set_title('NotCar in HLS')
axes[4, 1].imshow(hog_image_h)
axes[4, 1].set_title('Hog ch. H')
axes[4, 2].imshow(hog_image_l)
axes[4, 2].set_title('Hog ch. L')
axes[4, 3].imshow(hog_image_s)
axes[4, 3].set_title('Hog ch. S')

## CAR IMAGE: DRAW YCrCb HOG FEATURES
notcar_img_YCrCb = cv2.cvtColor(notcar_img, cv2.COLOR_RGB2YCrCb)
notcar_img_YCrCb_Y = notcar_img_YCrCb[:,:,0]
notcar_img_YCrCb_Cr = notcar_img_YCrCb[:,:,1]
notcar_img_YCrCb_Cb = notcar_img_YCrCb[:,:,2]

features, hog_image_Y = get_hog_features(notcar_img_YCrCb_Y, orient, pix_per_cell, cell_per_block,
                                         vis=True, feature_vec=True)
features, hog_image_Cr = get_hog_features(notcar_img_YCrCb_Cr, orient, pix_per_cell, cell_per_block,
                                          vis=True, feature_vec=True)
features, hog_image_Cb = get_hog_features(notcar_img_YCrCb_Cb, orient, pix_per_cell, cell_per_block,
                                          vis=True, feature_vec=True)

axes[5, 0].imshow(notcar_img_YCrCb)
axes[5, 0].set_title('NotCar in YCrCb')
axes[5, 1].imshow(hog_image_Y)
axes[5, 1].set_title('Hog ch. Y')
axes[5, 2].imshow(hog_image_Cr)
axes[5, 2].set_title('Hog ch. Cr')
axes[5, 3].imshow(hog_image_Cb)
axes[5, 3].set_title('Hog ch. Cb')


fig.tight_layout()
plt.savefig('../output_images/HOG.png')

####### Vary Orientation

pix_per_cell = 8
cell_per_block = 2

fig = plt.figure()
plt.rc('font', size=14)
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))

car_img_hls = cv2.cvtColor(car_img, cv2.COLOR_RGB2HLS)
car_img_hls_h = car_img_hls[:,:,0]
car_img_hls_l = car_img_hls[:,:,1]
car_img_hls_s = car_img_hls[:,:,2]

features, hog_image_l_1 = get_hog_features(car_img_hls_l, 7, pix_per_cell, cell_per_block,
                                           vis=True, feature_vec=True)
features, hog_image_l_2 = get_hog_features(car_img_hls_l, 9, pix_per_cell, cell_per_block,
                                         vis=True, feature_vec=True)
features, hog_image_l_3 = get_hog_features(car_img_hls_l, 12, pix_per_cell, cell_per_block,
                                           vis=True, feature_vec=True)
features, hog_image_l_4 = get_hog_features(car_img_hls_l, 15, pix_per_cell, cell_per_block,
                                           vis=True, feature_vec=True)

axes[0].imshow(car_img_hls)
axes[0].set_title('Car in HLS')
axes[1].imshow(hog_image_l_1)
axes[1].set_title('Ch. L, orient=7')
axes[2].imshow(hog_image_l_2)
axes[2].set_title('Ch. L, orient=9')
axes[3].imshow(hog_image_l_3)
axes[3].set_title('Ch. L, orient=12')
axes[4].imshow(hog_image_l_4)
axes[4].set_title('Ch. L, orient=15')

fig.tight_layout()
plt.savefig('../output_images/HOG_orient.png')


####### Vary Pixel per Cells

orient = 9
cell_per_block = 2

fig = plt.figure()
plt.rc('font', size=14)
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))

car_img_hls = cv2.cvtColor(car_img, cv2.COLOR_RGB2HLS)
car_img_hls_h = car_img_hls[:,:,0]
car_img_hls_l = car_img_hls[:,:,1]
car_img_hls_s = car_img_hls[:,:,2]

features, hog_image_l_1 = get_hog_features(car_img_hls_l, orient, 6, cell_per_block,
                                           vis=True, feature_vec=True)
features, hog_image_l_2 = get_hog_features(car_img_hls_l, orient, 8, cell_per_block,
                                           vis=True, feature_vec=True)
features, hog_image_l_3 = get_hog_features(car_img_hls_l, orient, 10, cell_per_block,
                                           vis=True, feature_vec=True)
features, hog_image_l_4 = get_hog_features(car_img_hls_l, orient, 12, cell_per_block,
                                           vis=True, feature_vec=True)

axes[0].imshow(car_img_hls)
axes[0].set_title('Car in HLS')
axes[1].imshow(hog_image_l_1)
axes[1].set_title('Ch. L, pix_per_cell=6')
axes[2].imshow(hog_image_l_2)
axes[2].set_title('Ch. L, pix_per_cell=8')
axes[3].imshow(hog_image_l_3)
axes[3].set_title('Ch. L, pix_per_cell=10')
axes[4].imshow(hog_image_l_4)
axes[4].set_title('Ch. L, pix_per_cell=12')

fig.tight_layout()
plt.savefig('../output_images/HOG_pix_per_cell,orient=8,cellblock=2.png')


####### Vary Cell per Block

orient = 9
pix_per_cell = 8

fig = plt.figure()
plt.rc('font', size=14)
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))

car_img_hls = cv2.cvtColor(car_img, cv2.COLOR_RGB2HLS)
car_img_hls_h = car_img_hls[:,:,0]
car_img_hls_l = car_img_hls[:,:,1]
car_img_hls_s = car_img_hls[:,:,2]

features, hog_image_l_1 = get_hog_features(car_img_hls_l, orient, pix_per_cell, 1,
                                           vis=True, feature_vec=True)
features, hog_image_l_2 = get_hog_features(car_img_hls_l, orient, pix_per_cell, 2,
                                           vis=True, feature_vec=True)
features, hog_image_l_3 = get_hog_features(car_img_hls_l, orient, pix_per_cell, 4,
                                           vis=True, feature_vec=True)
features, hog_image_l_4 = get_hog_features(car_img_hls_l, orient, pix_per_cell, 8,
                                           vis=True, feature_vec=True)

axes[0].imshow(car_img_hls)
axes[0].set_title('Car in HLS')
axes[1].imshow(hog_image_l_1)
axes[1].set_title('Ch. L, cell_per_block=1')
axes[2].imshow(hog_image_l_2)
axes[2].set_title('Ch. L, cell_per_block=2')
axes[3].imshow(hog_image_l_3)
axes[3].set_title('Ch. L, cell_per_block=4')
axes[4].imshow(hog_image_l_4)
axes[4].set_title('Ch. L, cell_per_block=6')

fig.tight_layout()
plt.savefig('../output_images/HOG_cell_per_block,orient=9,pix=8.png')

