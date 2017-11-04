# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[architecture]: ./architecture-diagram.png "Architecture"

[image1]: ./results/image1.jpg "Visualization"

[train1]: ./results/train-1.jpg "No Passing"
[train2]: ./results/train-2.jpg "Yield"
[train3]: ./results/train-3.jpg "Road Work"

[grayscaling1]: ./results/before-grayscaling.jpg "Before Grayscaling"
[grayscaling2]: ./results/after-grayscaling.jpg  "After Grayscaling"

[augment1]: ./results/augment1.jpg  "Data Augmentation 1"
[augment2]: ./results/augment2.jpg  "Data Augmentation 2"
[augment3]: ./results/augment3.jpg  "Data Augmentation 3"
[augment4]: ./results/augment4.jpg  "Data Augmentation 4"


[test1]: ./test/1.jpeg "Traffic Sign 1"
[test1-32x32]: ./test/1-processed.jpeg "32x32 Traffic Sign 1"
[test1-predictions]: ./test/1-predictions.jpg "Traffic Sign 1 Predictions"

[test2]: ./test/4.jpeg "Traffic Sign 2"
[test2-32x32]: ./test/4-processed.jpeg "32x32 Traffic Sign 2"
[test4-predictions]: ./test/4-predictions.jpg "Traffic Sign 1 Predictions"

[test3]: ./test/11.jpeg "Traffic Sign 3"
[test3-32x32]: ./test/11-processed.jpeg "32x32 Traffic Sign 3"
[test11-predictions]: ./test/11-predictions.jpg "Traffic Sign 1 Predictions"

[test4]: ./test/25.jpeg "Traffic Sign 4"
[test4-32x32]: ./test/25-processed.jpeg "32x32 Traffic Sign 4"
[test25-predictions]: ./test/25-predictions.jpg "Traffic Sign 1 Predictions"

[test5]: ./test/27.jpeg "Traffic Sign 5"
[test5-32x32]: ./test/27-processed.jpeg "32x32 Traffic Sign 5"
[test27-predictions]: ./test/27-predictions.jpg "Traffic Sign 1 Predictions"

[fm-1-input]: ./test/activations/1-input.jpg "32x32 Traffic Sign 1 Input"
[fm-1-conv1]: ./test/activations/1-conv1-feature-map.jpg "32x32 Traffic Sign 1 Conv1"
[fm-1-conv2]: ./test/activations/1-conv2-feature-map.jpg "32x32 Traffic Sign 1 Conv2"

[fm-4-input]: ./test/activations/4-input.jpg "32x32 Traffic Sign 1 Input"
[fm-4-conv1]: ./test/activations/4-conv1-feature-map.jpg "32x32 Traffic Sign 1 Conv1"
[fm-4-conv2]: ./test/activations/4-conv2-feature-map.jpg "32x32 Traffic Sign 1 Conv2"

[fm-11-input]: ./test/activations/11-input.jpg "32x32 Traffic Sign 1 Input"
[fm-11-conv1]: ./test/activations/11-conv1-feature-map.jpg "32x32 Traffic Sign 1 Conv1"
[fm-11-conv2]: ./test/activations/11-conv2-feature-map.jpg "32x32 Traffic Sign 1 Conv2"

[fm-25-input]: ./test/activations/25-input.jpg "32x32 Traffic Sign 1 Input"
[fm-25-conv1]: ./test/activations/25-conv1-feature-map.jpg "32x32 Traffic Sign 1 Conv1"
[fm-25-conv2]: ./test/activations/25-conv2-feature-map.jpg "32x32 Traffic Sign 1 Conv2"

[fm-27-input]: ./test/activations/27-input.jpg "32x32 Traffic Sign 1 Input"
[fm-27-conv1]: ./test/activations/27-conv1-feature-map.jpg "32x32 Traffic Sign 1 Conv1"
[fm-27-conv2]: ./test/activations/27-conv2-feature-map.jpg "32x32 Traffic Sign 1 Conv2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

link to my [project code](https://github.com/jazracherif/udacity_sdcnd/blob/master/CarND-Traffic-Sign-Classifier-Project/writeup.md)

---

### I- Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the numpy and matplotlib library to calculate summary statistics of the traffic signs data set and plot them:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32, 32, 3
* The number of unique classes/labels in the data set is 4

#### 2. Exploratory visualization of the dataset.

First, here are some 32x32x3 pictures taken randomly from the dataset. We  can see the haziness of the pictures, and how some of them need extra visual work to identify what the sign is saying.

![alt text][train1]
![alt text][train2]
![alt text][train3]

Here is an exploratory visualization of the training dataset. It is a histogram showing how the data is distributed amongst the various classes.

![alt text][image1]

We can see that the distribution of pictures amongst classes varies widely, some classes such as class 1:"Speed Limit (30Km/h)" and class 2:"Speed Limit (50Km/h)" have around 2000 pictures while classes like class 0:"Speed limit (20km/h)" and class 37:"Go straight or left" have less than 200 pictures

Here are the top-5 most common classes and 5 least common classes in terms of number of points
Most Common Labels
1)  2010 pts : Speed limit (50km/h) (2)
2) 1980 pts : Speed limit (30km/h) (1)
3) 1920 pts : Yield (13)
4) 1890 pts : Priority road (12)
5) 1860 pts : Keep right (38)

Least Common Labels
1) 180 pts : Speed limit (20km/h) (0)
2) 180 pts : Go straight or left (37)
3) 180 pts : Dangerous curve to the left (19)
4) 210 pts : End of all speed and passing limits (32)
5) 210 pts : Pedestrians (27)

### II- Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In order to meet the target of 93% accuracy on the Validation set, I only needed the following two preprocessing techniques:
1. Grayscaling
2. Normalization

As a first step, I decided to convert the images to **grayscale** because the color information did not really add much to the kind of label the image represented. Accuracy improved substantially after making this change.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][grayscaling1]
![alt text][grayscaling2]

As the second step, I **normalized** the image data because this will allow better convergence during Gradient descent optimization

**Data Augmentation Work**
In the beginning, when I was changing my architecture and could not meet the target accuracy score, I decided to use data augmentation, though I did not use it in the end product. I was inspired by the discussion in this post on [Traffic sign classification](https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc) and used the following techinques:
1. Brightness increase
2. Rotation
3. Translation
4. Shear

I spent a little time tuning the data set. Here are some examples I was able to generate:

![alt text][augment1]
![alt text][augment2]
![alt text][augment3]
![alt text][augment4]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the Lenet model with the addition of connected the first convolution layer just before pooling with the first fully connected layer, a technique taken from the [Lecun Paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

See the architecture below:

![alt text][architecture]

More detailed description in the following table:


Layer id | Layer         		|     Description	        					|
|:----|:---------------------:|:---------------------------------------------:|
|0 | Input         		| 32x32x3 RGB image   							|
|1| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
|2| RELU					|												|
|3| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
|4| Convolution 5x5	    |  1x1 stride, same padding, outputs 10x10x16       									|
|5| RELU          |                        |
|6| Max pooling          | 2x2 stride,  outputs 5x5x16         |
|7 | flatten | output of layer 2 and layer 6
|8| Fully connected		| input 5104, output 120       									|
|9| RELU				| .        									|
|10| Dropout		|	keep_prob =0.8											|
|11|	Fully Connected				|	120x84											|
|12|  Softmax        |  84*43                      |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following parameters
* learning_rate = 0.001
* Batch Size = 128
* Dropout = 0.8
* Optimizer = Adam
* n_epoch = 60

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

I started with the LENET model because it's a simple and proven architecture known to work on small pictures to detect letters and numbers. I first focused on tuning the learning rate as well as the initial variance values. For values of alpha= 0.001 and sigma= 0.1, I was able to get the accuracy on the traning set to be above 0.95, but the Validation accuracy was stuck at 0.89

I then attempt to follow some of the steps mentionned in the  [Lecun Paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) in particular connecting the convolutional later to the fully connected layer.  This immediately benefited the validation error as well as the training error, which could now reach 0.99.

Before reaching the above results, I also attempting changing the model, such as adding additional convolution layer as well as expanding the fully connected layer to contain more units, for example 1024x1024. I found that all these models overfit the training set but then did not do as well on the validation set. I therefore went back to the simpler LENET architecture and made the change mentionned above.

Finally, in order to deal with the variance problem and get the validation score up beyong the 0.93 threshold, I added a Dropout layer just after the first Fully Connected layer, with keep_prob = 0.8. This was enough to push the accuracy to 0.94

My final model results were:
* Training set accuracy of 0.986
* Validation set accuracy of 0.931
* Test set accuracy of 0.904

The training set accuracy is very high, meaning that the model is performing very well over the training set. The validation set accuracy is also high and meets the target for this exercise. The fact that there is a difference of 5.8% between the two, means that there is room for improvement as the system is overfitting a bit and there is somewhat high variation. The test set error is also lower than the validation set error by 2.4%, which is not a large difference. However the model may be overfitting a little bit on the validation set, a problem which may be solved by adding more points to the validation set.

As a next step, more can be done to increase the validation set accuracy such as data augmentation and regularization techniques.

### III- Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web, each of them of different size and next to its its processed version in 32x32 ratio:

Speed limit (30km/h)

![alt text][test1] ![alt text][test1-32x32]

Speed limit (70km/h)

![alt text][test2] ![alt text][test2-32x32]

Right-of-way at the next intersection

![alt text][test3] ![alt text][test3-32x32]


Road work

![alt text][test4] ![alt text][test4-32x32]

Pedestrians

![alt text][test5] ![alt text][test5-32x32]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| label | Image			        |     Prediction	        					|
|:----|:---------------------:|:---------------------------------------------:|
|1  | Speed limit (30km/h)      		| Speed limit (30km/h)   									|
| 4  | Speed limit (70km/h)          |  Bumpy Road (22)                      |
|11  | Right-of-way at the next intersection (11)     			| Right-of-way at the next intersection (11)								|
|25  | Road work			| Traffic signals (26)      							|
|27 | Pedestrians            | Go straight or left (37)                   |

The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This does not compare favorably to the accuracy on the test set, which is 0.916.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 49th cell of the Ipython notebook.

For the 1st image, Expected label is Speed limit (30km/h) (1), correct label predicted with very high probability 0.999

![alt text][test1-predictions]

| Probability | Label |
|:---------------------:|:-------------------------:|
| **0.999408** | **Speed limit (30km/h) (1)** |
| 0.000407973 | Speed limit (20km/h) (0) |
| 7.7941e-05 | Speed limit (80km/h) (5) |
| 6.44589e-05 | Speed limit (70km/h) (4) |
| 3.86881e-05 | Speed limit (50km/h) (2) |

For the 2nd image, Expected label is Right-of-way at the next intersection (11), correct label predicted with very high probability 0.998

![alt text][test11-predictions]

| Probability | Label |
|:---------------------:|:-------------------------:|
| **0.998492** | **Right-of-way at the next intersection (11)** |
| 0.00146018 | Beware of ice/snow (30) |
| 4.6978e-05 | Double curve (21) |
| 5.5106e-07 | Pedestrians (27) |
| 7.08638e-09 | General caution (18) |

For the 3rd image, Expected label is Speed limit (70km/h) (4), wrong label predict with a medium probability of 0.52, none of the top 5 values predict this label

![alt text][test4-predictions]

| Probability | Label |
|:---------------------:|:-------------------------:|
| 0.529152 | Bumpy road (22) |
| 0.230067 | Dangerous curve to the right (20) |
| 0.19349 | Road work (25) |
| 0.0369606 | Bicycles crossing (29) |
| 0.00258905 | Go straight or right (36) |

For the 4th image, Expected label is Pedestrians (27), wrong label predict with a medium probability of 0.49, none of the top 5 values predict this label

![alt text][test27-predictions]

| Probability | Label |
|:---------------------:|:-------------------------:|
| 0.496878 | Go straight or left (37) |
| 0.180581 | Stop (14) |
| 0.115086 | Speed limit (70km/h) (4) |
| 0.0638486 | Speed limit (30km/h) (1) |
| 0.0298649 | Traffic signals (26) |

For the 5th image, Expected label is Road work (25), we find the truth label in the top 5 list but with a very low probability

![alt text][test25-predictions]

| Prob | Label |
|:---------------------:|:-------------------------:|
| 0.390784 | Traffic signals (26) |
| 0.139406 | General caution (18) |
| 0.120089 | Pedestrians (27) |
| **0.0872249** | **Road work (25)** |
| 0.0701883 | Road narrows on the right (24) |

### Visualizing the Neural Network
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Speed limit (20km/h) (1)

![alt text][fm-1-input]

![alt text][fm-1-conv1]

![alt text][fm-1-conv2]


Speed limit (70km/h) (4)

![alt text][fm-4-input]

![alt text][fm-4-conv1]

![alt text][fm-4-conv2]


Right-of-way at the next intersection (11)

![alt text][fm-11-input]

![alt text][fm-11-conv1]

![alt text][fm-11-conv2]


Road work (25)

![alt text][fm-25-input]

![alt text][fm-25-conv1]

![alt text][fm-25-conv2]


Pedestrians (27)

![alt text][fm-27-input]

![alt text][fm-27-conv1]

![alt text][fm-27-conv2]
