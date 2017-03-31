# Behavioral Cloning Project
## Goals and Steps
The goals of this project (to me) were the following:
* Use the simulator to collect data of good driving behavior
* Experiment with different architectures and parameters in Keras
* Experiment with AWS S3 and EC2 GPU instances

The steps of this project are the following:
* Use the training data provides in the resources
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Model_Visualization.png "Model Visualization"
[image2]: ./center_image.jpg "Center Camera Image"
[image3]: ./left_image.jpg "Left Camera Image"
[image4]: ./right_image.jpg "Right Camera Image"
[image5]: ./flipped_image.jpg "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files in Yaser_Eftekhari_Proj3.zip:
* model.py: the script to create and train the model
* drive.py: to drive the car in autonomous mode
* model.h5: a trained convolution neural network
* run_10.mp4: video of the car driving in autonomous mode in track 1 with speed set to 10 mph
* run_30.mp4 video of the car driving in autonomous mode in track 1 with speed set to 30 mph
* writeup_report.md: this report summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. The script can takes the following input arguments:
- number of epochs (default to 5): --epochs
- relative path to the training data (default to "data/"): --path
- correction factor for adjusting the steering wheel angle for left and right images (default to 0.2): --correction

A sample of how to use the script is as follows:

```sh
python model.py --epochs 5 --correction 0.1
```

---
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a the following layers in order (model.py lines 65-83):
- Normalization layer using a Keras lambda layer
- Cropping layer
- 3 Convolutional layers with filter size 5x5 and depths 24, 36, and 48 with RELU activation
- 2 Convolutional layers with filter size 3x3 and depth 64 with RELU activation
- Flattening layer
- 3 Fully connected layers of size 100, 50 and 10 followed by the output layer

#### 2. Attempts to reduce overfitting in the model

I also experimented with this architectures by adding drop outs and it still worked fine (although I added the dropout layer after all layers convolution and fully connected with dropping probability of 0.2). This part is not submitted.

I monitored the training and validation accuracies as a function of number of epochs and found out 3 is a good compromise.

The model was trained and validated on random shuffles of the data set to ensure that the model was no bias in training (code line 89).

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 86). However, the number of epochs and the correction factor for the left and right images where optimized through trial and error. Ultimately 3 and 0.15 where chosen as best number of epochs and correction factor.

#### 4. Appropriate training data

I used the training data provided in the resources. As the model performed well on them, there was no need to extra training data.

The only modification to the training data was to convert them from BGR to YUL. However, even without this modification the model performs well and does not go out of bounds.

For details about how I created the training data, see the next section.

---
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a good known architecture and tune the parameters.

My first step was to use the architecture proposed in the Nvidia paper. I thought this model is appropriate because it had a mixture of convolution layers and fully-connected layers. Also as the paper claimed they achieved good results on tracks with no marking which was similar to the simulation track used for this project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set by a ratio of 5:1.

I ran many simulations with different number of epochs and correction factors. I found that the model overfitted on 5 epochs as the MSE increased compared to the case of 3 epochs.

Simulation also revealed that a correction factor of 0.2 is too high and 0.1 is too low. So I chose 0.15 as a good number in between.

I also experimented with using the BGR output of imread or to convert them to YUL (as suggested by the Nvidia paper).

It is worth mentioning that the model drives fine with all such parameters without going out of bounds. The main consideration was the swerving of the car while driving and how close it got to the road shoulders.

After each simulation I ran the simulator to see how well the car was driving around track one.

One major change I had to do was to change the way drive.py reads and passes the images to the model. CV2 imread is used to prepare the images for training and it outputs the images in BGR. However, the code in drive.py reads the images as RGB. So, I had to twick drive.py to follow the same steps of the training moodel.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 65-83) consisted of the following layers, kernel sizes and depths:
- Normalization layer using a Keras lambda layer
- Cropping layer
- 3 Convolutional layers with filter size 5x5 and depths 24, 36, and 48 with RELU activation
- 2 Convolutional layers with filter size 3x3 and depth 64 with RELU activation
- Flattening layer
- 3 Fully connected layers of size 100, 50 and 10 followed by the output layer

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

The training is performed on center, left and right images. In addition to these provided images, each image is also flipped and added to the training set. Needless to say that the steering wheel angles have been adjusted.

Snapshots below show the center, left and right images.

![alt text][image2]

![alt text][image3]

![alt text][image4]

This is also the flipped image of the center camera.

![alt text][image5]

After the collection process, I had 24,108x2=48,216 number of data points. The images are converted from BGR to YUL, normalized (model.py line 67), and then cropped 70 pixels from top and 25 from bottom (model.py line 69).

I finally randomly shuffled the data set and put 20% of the data into a validation set (model.py line 89).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
