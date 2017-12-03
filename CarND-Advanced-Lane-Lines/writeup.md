# Advanced Lane Finding Project

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calib]: ./output_images/calibration.jpg "calibration Image"
[undistorted]: ./output_images/undistorted.png "Undistorted"
[pipeline-out]: ./output_images/pipeline-out.png "Pipeline Out"
[warped]: ./output_images/warped.png "Warped picture"
[histogram]: ./output_images/histogram.png "Histogram"
[sliding-window]: /output_images/sliding-window.png "Sliding Window"
[scatter]: /output_images/scatter.png "Scatter plot"
[lane-window]: /output_images/lane-window.png "Lane window"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file calibrate.py located in "./code/"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world.

Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][calib]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The threshold and color transformation functions pipelines are done in "pipelines.py" and "gradient_threshold.py".

The function pipeline() (pipelines.py line 41) runs the undistorded image into both a color and a gradient threshold transformation before combining them:

Color Threshold Transformation (pipelines.py: line 53 - 62):
1. Transform the undistored image from RGB to HLS
2. **yellow_binary**: the output of a color threshold on the H channel, done to capture the yellow line. (H=19)
3. **l_binary**: the output of a thresholding on the L channel to keep lighter parts of the picture, filtering anything below 160.

Gradient Threshold transforms (combined_thresh() in gradient_threshold.py: line 133):
1. Run a sobel operation on both x and y axis with kernel size 5 and threhols 20,100
2. Apply a magnitude thresholding on both x and y axis with thresholds 30,100
3. Apply a direction of gradient thresholding with angle threshold between 0.7 and 1.3
4. **sxbinary**: Combine all 3 previous operation with an OR operations

I combine the output of these two transformations in the following way:
1. Use the right part of the Gadient Threshold and AND it with the l_binary output
2. Add the result with the yellow_binary image which capture the left yellow line.

`combined_binary[(yellow_binary==1) | ((l_binary==1) & (sxbinary_right==1))] = 1`

![alt text][pipeline-out]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform appear in line 194-201 in main.py. I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([ [550, 470], [800, 470], [1150, 710],  [180, 710]])
dst = np.float32([[0, 0],[1280, 0], [1280, 720], [0, 720]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane identication logic is implemented in file "/code/lane_detection.py" and is called from within the process_image() function at lines 215-226

2 additional classes were defined to help with this:

a  Line class to help maintain past information about the left and right lines that forms the lane:
```python
class Line()  #(main.py:line 90)
```
a class that implements exponential moving average to be applied to each of the lines polynomial coeffcients (a,b, and c in a.x^2 + b.x +c)
```python
class movingAvg() #(main.py:line 11)
```
The Lane Detection logic is divided into 2 parts:
1. For the first image or whenever a problem is detected with the previous image, a call to find_lanes() (line 117) is made. This function takes as input the warped image and then applies the following logic:
    1. If previous lane data is available, generate a set of point that fit the line and add that to the warped picture. This create a simple continuity and availablility of data.
        ```python
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        fitx = (current_fit.a*ploty**2 + current_fit.b*ploty + current_fit.c).astype(int)
        ```
    2. Create a histogram of the number of nonzero pixels at each x location, and find the leftmost and rightmost peaks. These peaks will correspond to the left and right line of the current lane. See an example below:

        ![alt text][histogram]
    
    3. Run a vertical sliding window around each peak capturing the nonzero pixels around a margin of 120px from the center, and recentering the windown whenver it contains more than 50 points. There are a total of 9 windows each of 720/9=80px height. See an example of this below:

        ![alt text][sliding-window]

    4. Fit a polynomial line of degree 2 to the points detected for each line.
    5. Update the line's equation coefficients using an exponential moving average with beta=0.5
        ```python
        self.a = (self.beta_a * self.a + (1 - self.beta_a) * a )
        self.b = (self.beta_b * self.b + (1 - self.beta_b) * b )
        self.c = (self.beta_c * self.c + (1 - self.beta_c) * c )
        ```
    6. Store the line information using function update_line_info() (lane_detection.py: line 27)
    
2. In case a lane has been detected, don't run the exhaustive window sliding logic of 1.III but instead search for points close to the previously detected line. The funtion process_next_image() (lane_detection.py: line 258) is called instead and the following steps are executed:
    1. Retrieve each lane's fitted equations and filter points which are within a 100pixel margin from the line's center. The green region in the picture below illustrate the searched window.
    
        ![alt text][lane-window]

    2. To the detected points add an additional set which correspond to the line's equation in order to add robustness (similar to step 1-I).
    3. Concatenate the last 10 sets of points detected and fit a polynomial of degree 2. This is done in update_line_fit() (lane_detection.py: line 7). Below is a scatter plot of the left and right lines captured at the 4th image of the project_video.mp4 file:
    
        ![alt text][scatter]
        
    4. Update the current lane coefficient using exponential moving average
    

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
