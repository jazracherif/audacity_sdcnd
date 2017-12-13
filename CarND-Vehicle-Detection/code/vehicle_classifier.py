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


# Reduce the sample size because HOG features are slow to compute
#sample_size = 100
#cars = cars[0:sample_size]
#notcars = notcars[0:sample_size]

colorspace = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

spatial_size = (64, 64) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = False # Spatial features on or off

hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


## 1. LOAD THE DATA
use_cache = True
if use_cache and os.path.exists("./data.pkl"):
    # Load the cached Features
    
    data = pickle.load(open("./data.pkl", "rb"))
    X_train, X_test, y_train, y_test = data["X_train"],  data["X_test"],  data["y_train"], data["y_test"]
    X_scaler = joblib.load('./vehicle_classifier_scaler.pkl')

else:
    
    t=time.time()
    car_features = extract_features(cars, color_space=colorspace, orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel=hog_channel,spatial_size=spatial_size,hist_bins=hist_bins,
                                    spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    notcar_features = extract_features(notcars, color_space=colorspace, orient=orient,
                                       pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_size=spatial_size,hist_bins=hist_bins,
                                       spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    t2 = time.time()

    print(round(t2-t, 2), 'Seconds to extract HOG features...')
    
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    # Save Scaler
    joblib.dump(X_scaler, 'vehicle_classifier_scaler.pkl')

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    #X_train, X_test, y_train, y_test = train_test_split(
    #                                                    scaled_X, y, test_size=0.2, random_state=rand_state)

    X_train, X_test, y_train, y_test = train_test_split(
                                                    scaled_X, y, test_size=0.2)

    data = {"X_train":X_train, "X_test":X_test, "y_train":y_train, "y_test":y_test}
    pickle.dump(data, open("./data.pkl", "wb+"))


## 2. START TRAINING PROCEDURE
print('Using:',orient,'orientations',pix_per_cell,
      'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

do_LinearSVC = True
do_svc_grid = False
do_svc_VotingClassifier= False
do_GradientBoostingClassifier = False

if do_LinearSVC:
    model = LinearSVC(C=0.1, verbose=True, max_iter=1000)

    t=time.time()
    model.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

elif do_svc_grid:
    parameters = {'kernel':['linear'], 'C':[1e-5, 1e-04, 1e-3, 1e-3]}
#    parameters = { 'C':[1e-5, 1e-04, 1e-3, 1e-3]}

    svc = svm.SVC(probability=True)

    clf = GridSearchCV(svc, parameters)
    results = clf.fit(X_train, y_train)
    
    print ("results", results.cv_results_)
    model = results.best_estimator_

elif do_svc_VotingClassifier:
    svc = VotingClassifier(estimators=[
                    ('clf1', LinearSVC(C=10)),
                    ('clf2', LinearSVC(C=0.01)),
                    ('clf3', LinearSVC(C=0.001)),
                    ('clf4', LinearSVC(C=0.1)),
                    ('clf5', LinearSVC(C=1))
                   ], voting='hard')
    clf_estimators = []
    for i in [ 10**(j) for j in range(-8,1, 1)]:
        clf_estimators.append(('clf'+str(i), LinearSVC(C=i)))
    
    model = VotingClassifier(estimators=clf_estimators, voting='hard')

    t=time.time()
    model.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

elif do_GradientBoostingClassifier:
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                 max_depth=3, random_state=0).fit(X_train, y_train)

    t=time.time()
    model.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')


## 3. EVALUATE ON THE TEST SET
print ("Best Model Selected", model)

# Check the score of the model
print('Test Accuracy of the Model = ', round(model.score(X_test, y_test), 6))

# Check the prediction time for a single sample
t=time.time()

n_predict = 10
print('My model predicts: ', model.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels')

joblib.dump(model, 'vehicle_classifier.pkl')
joblib.dump(X_scaler, 'vehicle_classifier_scaler.pkl')


