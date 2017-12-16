# Vehicle Detection Project

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[car_not_car]: ./output_images/car_not_car.png ' Car and notCar'
[hog]: ./output_images/HOG.png 'Hog Features'
[hog_orient]: ./output_images/HOG_orient.png 'Hog Features'
[hog_pix_per_cell]: ./output_images/HOG_pix_per_cell,orient=8,cellblock=2.png 'Hog Features'
[hog_cell_per_block]: ./output_images/HOG_cell_per_block,orient=9,pix=8.png 'Hog Features'


[windows1]: ./output_images/image1-scale-0.85-cells-3.jpg 'Windows 1'
[windows2]: ./output_images/image1-scale-1.25-cells-2.jpg 'Windows 2'
[windows3]: ./output_images/image1-scale-1.5-cells-3.jpg 'Windows 3'
[windows4]: ./output_images/image1-scale-2.25-cells-2.jpg 'Windows 4'


[seq_image1]: ./output_images/seq_image1.jpg'Seq Image 1'
[seq_image2]: ./output_images/seq_image2.jpg 'Seq Image 2'
[seq_image3]: ./output_images/seq_image3.jpg 'Seq Image 3'
[seq_image4]: ./output_images/seq_image4.jpg 'Seq Image 4'

[seq_image4_label]: ./output_images/seq_image4-label.jpg 'Seq Image 4 label'
[seq_image4_final]: ./output_images/seq_image4-final.jpg 'Seq Image 4 final'


[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

List of relevant files:
- `feature_extraction_utilities.py`: Utility function to extract HOG features
- `vehicle_classifier.py`:  Implements various methods for training a vehicle classifier based on HOG, color histogram, and spatial binning.
- `nn_classifier.py` : Implements a method to learn convolutional neural network to classify pictures as vehicle or non-vehicles.
- `vehicle_detection.py`: Code for detecting vehicle in a picture, and drawing bounded box on a video stream.
- `visualizations.py`: code for creating HOG Features visualizations.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the file `feature_extraction_utilities.py` and is inspired by code from section 29 HOG Classify for this project. The function **extract_features** (`feature_extraction_utilities.py, line:57)`  takes in the image and parameters for performing a HOG feature extraction, as well as a color histogram and a spatial binning. **get_hog_features()** (line17) eventuall calls **skimage.feature.hog**

Before extracting the features, I started by reading in all the [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images from the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][car_not_car]

I  explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). Here is a comparison of both Car and notCar images using the `RGB`, `HLS`, and `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog]

In the Car case, the RGB channels capture the shape of the vehicle nicely, however all 3 channels are the same while the other encoding capture different kind of information. Channel L of the HLS seem to capture best the shape of the vehicle while its channel S seems to bring out the rear of the vehicle. Similarly, Channel Y of YCrCb also performs well on the overall shape. In this project, I opted for a combination of the 3 HLS channels to capture as much distinct information about the car

In the following series, we look at varying the orientation, pixels_per_cell, and cells_per_block parameters for the HLS channel L image:

*pix_per_cell = 8, cell_per_block =2*

![alt text][hog_orient]

Orientation values seems to perform already well with 7 and 9 angles, and so a value of 9 was picked.

*orient=8, cell_per_block=2*

![alt text][hog_pix_per_cell]

With pix_per_cell values above 8, the car boundaries are lost while for values of less than 8, too much information seems to be captured. A value of 8 was picked.

*orient=8, pix_per_cell=8*

![alt text][hog_cell_per_block]

Strangely all values are 
#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using in file `vehicle_classifier.py (lines: 118-193)`.
- I have divided the data set in a training, validation and test sets each with a 20% split and randomization.
- I have tried several kinds of classifiers and strategies including the following
    - A grid search using LinearSVC over C with values between 1e-5, to 1.
    - A grid search over svm.SVC which uses libsvm and is thus slower to train.
    - A VotingClassifier with hard voting threshold over multiple LinearSVC models
    - A gradient Boosting classifier to compare with
    
The input features consisted of the set of HOG features over the HLS image, together with a color histogram using function extract_features (`vehicle_classifier.py (lines: 81-86)`). The parameters for HOG were:
- colorspace = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
- orient = 9
- pix_per_cell = 8
- cell_per_block = 2
- hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

The model that perform best was a LinearSVC with C=0.1
**Test Accuracy of SVC =  0.986486**
The final SVM model was stored in file vehicle_classifier.pkl and the scaler in vehicle_classifier_scaler.pkl

In addition to the SVM model, I have implemented and trainined a convolutional neural network using Keras to see how it performs using only the raw image pixels. I implemeted a simple architecture in `nn_classifier.py, see nn_architecture() line 116`, which consists of 2 Convolutional layers with maxpooling and 1 dropout layers, followed by Fully Connected Layer with 32 node and a logistic function. With this model, I was able to achieve the following results:
Training Accuracy: 0.9919
Validataion Accuracy: 0.9944
**Test accuracy: 0.994369369369**
The final NN model was stored as keras_cifar10_trained_model.h5

In my final solution, I combined both models in order to improve the overall detection accuracy.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First I implemented a sliding window search as described in the lecture videos but I eventually resorted to using the single HOG transformation over the whole image, followed by segmentation of the hog features into equivalent 64x64 image features. This was done in `vehicle_detection.py, find_cars() line:24`. The logic is borrowed from lesson 35 Hog subSampling Window search, and essentially iterates overall all blocks of hog features which corresponding to a 64x64 sub image, considering 2 additional parameters that can be set, the *scale* and the *cells_per_step*. For this project, I experimented with multiple values for *scale* randing from 0.6 to 2.5 and for *cells_per_step* from 2 to 9.

Here are some images of some of the bounding boxes that were drawn for the same image.

**scale = 0.85, cells_per_step=3**

![alt text][windows1]

**scale = 1.25, cells_per_step=3**

![alt text][windows2]

**scale = 1.5, cells_per_step=2**

![alt text][windows3]

**scale = 2.25, cells_per_step=2**

![alt text][windows4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched looked at both HLS and YCrCb multiple channel HOG features plus histograms of color in the feature vector, which provided a nice result.  By combining the SVM classified with a Neural Network, I was able to get a better classification outcome. Here are some example images:

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are 4 frames and their corresponding heatmaps:

![alt text][seq_image1]

![alt text][seq_image2]

![alt text][seq_image3]

![alt text][seq_image4]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![alt text][seq_image4_label]

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][seq_image4_final]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

