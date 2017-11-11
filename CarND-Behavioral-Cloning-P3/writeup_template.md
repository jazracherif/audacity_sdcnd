# **Behavioral Cloning**

---


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./draw.io-behavioral-cloning.jpg "Model Architecture"

[center1]: ./results/center1.jpg "Center Image 1"
[sharp-center]: ./results/sharp-turn-center_2017_11_08_23_59_27_590.jpg "Sharp Turn Center Image"
[sharp-left]: ./results/sharp-turn-left_2017_11_08_23_59_27_590.jpg "Sharp Turn Left Image"
[sharp-right]: ./results/sharp-turn-right_2017_11_08_23_59_27_590.jpg "Sharp Turn Right Image"

[center2]: ./results/center2.jpg "Center Image 2"
[center2-flipped]: ./results/center2-flipped.jpg "Center Image 2 flipped"

[hist1]: ./results/hist_raw_data.jpg "Data Histogram 1"
[hist2]: ./results/hist_rebalanced.jpg "Data Histogram 2"
[loss1]: ./results/loss1.jpg "Loss Function 1"
[loss2]: ./results/loss2.jpg "Loss Function 2"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py: the script to create and train the model
* drive.py: driving the car in autonomous mode
* model.h5:  a trained convolution neural network
* writeup_report.md: summarizes the results
* video.mp4:  a recording of 1 lap using the model in model.h5
* visualizations.html: contains downsampling and visualization code

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

There are few parameters that can be used to accelerate model exploration and evaluation:
* Number of epochs: How many epochs to run the training for
* Batch Size: How big should the batch size be
* Learning Rate: What learning rate to use for the optimizer.

The **resume** parameter allows me to resume training a model where I last left it off. This allows me to continue training in some cases where improvement seems possible, without starting from scratch.

The **test** parameter allows me to run the training logic on a smaller dataset and do sanity checks.

The **predict** parameter allows me to only run predictions on a small dataset for sanity check of the code that would be used in the drive.py file

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My final model is inspired from the Keras [MNIST example model](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py). (see create_mnist_mode() in model.py line 167)

I start up by resizing the input into a 64x64 image. I then normalize each picture to have the input centered at 0 and between -0.5 and 0.5.This is followed by 2 convolutional layers and 1 MaxPooling layer. The output is  flattened and run thorugh a fully conntected layer of size 50, followed by another dense layer with output of 1 and no activation. I applied dropout after the 2nd convolutional layer and the first fully connected layer.

#### 2. Attempts to reduce overfitting in the model

* The model contains dropout layers in order to reduce overfitting (model.py lines 223).
* The model was also trained and validated on different data sets to ensure that the model was not overfitting (code line 58-85).
* Data augmentation techniques were used, such as flipping, and using side immages, as well as generating additional data points around some weak track spots.
* Finally, the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

* The model used the Adam optimizer and the loss function is the Maximum Square Error (MSE) (model.py line 317).
* The learning rate was tuned to achieve faster convergence. Values ranging from 0.01 to 0.000001 were tried to reduce both the training and the validation loss as well as the Mean Square Error. The final value for the learning rate was 0.000001 and for batch size 128.
* Various initialization functions were also tried for the model parameters, amongst them were the "gloriot_normal" and "he_normal"".
* Further the number the number of epochs and the batch size were tuned.

#### 4. Appropriate training data

The Training data consists of data provided for the project as well as data I acquired through unity to better capture certain scenarios. Pre-processing of the data was done to remove bias toward driving straight.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple model and make it more complex.

After having validated the overall software flow with a simple fully connected layer, I started experimenting with various architectures. I attempted different stackings of convolutions layers followed by fully connected layers. I focused mostly on tuning the learning rate, reducing my loss, and then evaluating the model in Unity's autonomous mode. The two architectures I ended up with are the [MNIST example](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py) and the [NVIDIA end-to-end model](https://arxiv.org/pdf/1604.07316.pdf). My final model architecture was the simpler of the two which achieved the goal of this assignment, that is driving track 1 satisfactorily.

Initially, I was working on the raw dataset using just the center image and without data augmentation, and the car would not be able to handle turns. Making the model more complicated did not help with this problem. I therefore resorted to various data augmentation techniques:
1. Every image in the training set was flipped and its steering angle was negated
2. The left and right images were also included with the steering corrections such that the left image would need a higher angle and the right image a small angle. A correction value of 0.4 was used

After including these techniques, I could see the car properly handling the first turn in track 1 and getting on the bridge successfully. All worked fine until a second sharp left turn is reached, at which point the car kept going straight into the wilderness.

Making the model more complex did not help with this. After a few tries, I finally realized that perhaps I needed more data about sharp turns. I therefore got some more training data and after retraining I could see the car now attempting to turn at the second sharp turn. However, the car would still hit the sides and not finished the turn.

After visualizing the data, I notice a very large number of training points with zero steering angles. See the image below for the data histograms:

![alt text][hist1]

I decided to downsample the training set by randomly removing 3/4 of the data points with zero steering angle, all else remaining the same. The code for this is in the file visualizations.ipynb and the results in visualizations.html. The following plot shows the results after this operation:

![alt text][hist2]

After downsampling the dataset and then retraining with the MNIST model, I got much better performance.

In order to squeeze as much out of the model and run the training faster, I resized each image in the input to 64x64 before feeding it to the normalization step. In order to load the images efficiently, I used a generator function (generator p27) and did all the pre-processing inside the Keras TensorFlow pipeline rather than outside, thus taking advantage of the GPU speedup.

I got the following loss functions for the training and validation set after 50 epochs

![alt text][loss1]

As I could see that both validation and training error were going down, I continue the training for another 10 steps thank to the "resume" parameter I had implemented. The next plots show the additional decreases in the loss and a plateau-ing for both indicating this was a good stopping point.

![alt text][loss2]

I then proceeded to test this model in Unity where the car successfully drove through the whole track without getting off the road!


#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of the following layers:

| Layer |  Description |
|:---|:------------------|
| 1 |   Resize the input into a 64x64 image. |
|2  |   Normalize each picture around 0 and between -0.5 and 0.5  |
|3  |   Add a Convolution layer with 5 kernels of size 3x3  |
|4  |   Add a Convolution layer with 10 kernels of size 3x3 |
|5  |   Add a MaxPooling layer with pool size 2 |
|6  |   Add a Dropout layer with dropout=0.25   |
|7  |   Flatten the output |
|8. |   Add a Fully Connected layer of size 50  |
|9  |   Add a Dropout layer with dropout=0.5    |
|10|   Add a Fully Connected layer with output size 1   |

Here is a visualization of the architecture:

![alt text][model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the Udacity dataset and recorded one additional lap on track 1 using center lane driving. Here is an example image of center lane driving:

![alt text][center1]

I also recorded the vehicle taking sharp turns a few times. Here is one capture of the left, center and right turn:

![alt text][sharp-left]
![alt text][sharp-center]
![alt text][sharp-right]

To augment the data sat, I also flipped images and angles thinking that this would improve steering away from the road edges.  For example, here is an image that has been flipped:

**Original Image**

![alt text][center2]

**Flipped Image**

![alt text][center2-flipped]

After the collection process, including the downsampling, I had 5,204 images. I  randomly shuffled the data set and put 20% of the data into a validation set. When counting the flipped and side images, my total training dataset has 18,732 images.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 60 as evidenced by plotted loss function.
