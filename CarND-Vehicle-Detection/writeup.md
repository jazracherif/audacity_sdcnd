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

With pix_per_cell values above 8, the car boundaries are lost while for values of less than 8, too much information seems to be capture. A value of 8 was picked.

*orient=8, pix_per_cell=8*

![alt text][hog_cell_per_block]

Strangely all values are 
#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

