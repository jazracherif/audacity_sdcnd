# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[solidWhiteCurve]: ./test_images_out/solidWhiteCurve.jpg "Result"
[solidWhiteRight]: ./test_images_out/solidWhiteRight.jpg "Result"
[solidYellowCurve]: ./test_images_out/solidYellowCurve.jpg "Result"
[solidYellowCurve2]: ./test_images_out/solidYellowCurve2.jpg "Result"
[solidYellowLeft]: ./test_images_out/solidYellowLeft.jpg "Result"
[whiteCarLaneSwitch]: ./test_images_out/whiteCarLaneSwitch.jpg "Result"

---

### Reflection

### 1. Description of the pipeline

My pipeline consisted of 6 steps:
1. Extract the Grayscale version of the image
2. Perform a Gaussian Blur with a kernel
3. Run the Canny Detection Algorithm
4. Cut to the region of interest based on the height and width values
5. Run the Hough Transform
6. Final Cut of the region of interset.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

Please Find below the example pipeline for each of the pictures:

#solidWhiteCurve.jpg
![alt text][solidWhiteCurve]  

#solidWhiteRight.jpg
![alt text][solidWhiteRight]  

#solidYellowCurve.jpg
![alt text][solidYellowCurve]  

#solidYellowCurve2.jpg
![alt text][solidYellowCurve2]  

#solidYellowLeft.jpg
![alt text][solidYellowLeft]  

#whiteCarLaneSwitch.jpg
![alt text][whiteCarLaneSwitch]  


### 2.  Potential shortcomings with current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
