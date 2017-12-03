## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

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


[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

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

Color Transoforms (line 53 - 62 pipelines.py):
1. Transform the undistored image from RGB to HLS
2. **yellow_binary**: the output of a color threshold on the H channel, done to capture the yellow line. (H=19)
3. **l_binary**: the output of a threholding on the L channel to keep lighter parts of the picture, filtering anything below 160.

Gradient Threshold transforms (combined_thresh() in gradient_threshold.py line 133):
1. Run a sobel operation on both x and y axis with kernel size 5 and threhols 20,100
2. Apply a magnitude threholding on both x and y axis with thresholds 30,100
3. Apply a direction of gradient thresholding with angle threshold between 0.7 and 1.3
4. **sxbinary**: Combine all 3 previous operation with an OR operations

I combine the output of these two transformations in the following way:
1. Use the right part of the Gadient Threshold and AND it with the l_binary output
2. Add the result with the yellow_binary image which capture the left yellow line.

`combined_binary[ (yellow_binary==1) | ( (l_binary==1) & (sxbinary_right==1))] = 1`

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

The lane identication logic is implemented in file "/code/lane_detection.py". The logic is divided into 2 path:
1. When the data is new or there is a problem with the previous detection logic, a call to find_lanes() (line 117) is made. This function takes as input the warped image and then applies the following logic:

a) If previous lane data is available, generate a set of point that fit the line and add that to the picture. This create a simple continuity and availablility of data
`ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
fitx = (current_fit.a*ploty**2 + current_fit.b*ploty + current_fit.c).astype(int)`

b) Create a histogram the number of pixels points at each x location, and find the leftmost and rightmost peaks. These peaks should correspond to the left and right line of the current lane. See an example below:

![alt text][histogram]

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
