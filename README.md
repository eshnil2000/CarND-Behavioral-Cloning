# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, I use deep neural networks and convolutional neural networks to clone driving behavior,   train, validate and test a model using Keras. The model  output's a steering angle to an autonomous vehicle.

This project uses a simulator provided by Udacity https://github.com/udacity/self-driving-car-sim where you can steer a car around a track for data collection. I used image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

# Files & Code Quality
There are five key files: 
* [model.ipynb](./model.ipynb) (script used to create and train the model)
* [drive.py](./drive.py)drive.py (script to drive the car )
* [model.h5](./model.h5)(a trained Keras model)
* [run1.mp4](./run1.mp4) (a video recording of the vehicle driving autonomously around the track for at least one full lap)

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md


## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

Node:

On my first iteration, I tried using a basic fully connected neural network as a default baseline starting point to get the data feed, pre-processing, training, output/ validation flows working.

#### 2. Submission includes functional code Using the Udacity provided simulator and my drive.py file; the car can be driven autonomously around the track by executing

```
Python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.ipynb [model.ipynb](./model.ipynb) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My initial approach was to use a fully connected neural network, but I quickly switched to a model used by Comma.ai at https://github.com/commaai/research/blob/master/train_steering_model.py, as my car was drifting off of the driving track.

A model summary is as follows:

```
### INITIALIZE Keras sequential model
model = Sequential()

### Pre-process the data, center the data
model.add(Lambda(lambda x:x/127.5-1,input_shape=(160,320,3) ))

### Crop the images to remove extraneous information, focus on the immediate road ahead
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))

### USE THE MODEL DEFINED IN COMMA.AI Steering model
#https://github.com/commaai/research/blob/master/train_steering_model.py
##3 convolution layers interleaved with 3 Exponential Linear Units
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))

### FLatten the model, convert 3 dimensional output on Convolution layer to 1 dimension
model.add(Flatten())

### Add a dropout layer to prevent overfitting
model.add(Dropout(.2))
model.add(ELU())

### Add a Dense layer with 512 outputs
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())

### FInally reduce the output to 1 representing steering angle
model.add(Dense(1))
###END COMMA.AI MODEL

### COMPILE USING ADAM OPTIMIZER, SO THAT LEARNING RATE DOESNT HAVE TO BE SET MANUALLY
model.compile(optimizer="adam", loss="mse")
```

#### 2. Attempts to reduce overfitting in the model

I split my sample data into training and validation data. Using 80% as training and 20% as validation.
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data & training strategy

I trained the vehicle by running it for 2 laps on the First course, and 1 lap over the 2nd course. This was not sufficient to keep the car within the boundaries of the course. The vehicle kept wandering off at 3 distinct points: 
1. When it found a white/ grey edge on the road.
2. When it found a black edge on the road, on the bridge.
3. When it approached brown/ sandy areas which were not clearly marked.
4. When it happened to line up on a white edge.

The basic training on the laps was good enough to keep the vehicle within the track initially and then when it detected the red and white markings at sharp curves.

TO fix the wandering problem in the 3 mentioned areas, I trained the vehicle by recording recovery manouvers by showing it to take sharp steering angles away from the white, grey and black edges. 
I also trained the vehicle to take sharp recovery angles when it landed on sandy/ brown patches.
Finally, I trained the vehicle to start on the white boundaries and then recover inwards towards the tracks.

An example of training the vehicle to take sharp steering angles away from boundaries is show here [return.jpg](./return.jpg)
![Return](https://raw.githubusercontent.com/eshnil2000/CarND-Behavioral-Cloning/master/return.png)
In addition to these training strategies, I also augmented the data by flipping the images so it could learn behavior for both steering directions.

An example of the captured image and flipped image can be found here [flip.jpg](./flip.jpg) and [flipped.jpg](./flipped.jpg)

In crossing the bridge, it helped to train the vehicle to stay away from the black edges by weaving an "S" curve across the bridge to train it to take medium steering angles to recover away from black edges.

A key part of the training strategy involved using the generator function in keras/ python. Once significant amount of images had been collected, my laptop was running out of memory to load all the images in memory [8GB Mac PRO]. Instead of switching to a more powerful machine on AWS or switching to a GPU instance, I used generators to load images as and when needed and available. This allowed me to complete the training process smoothly.

### Keras code: 
```
#setup generators, feed data in batches of 32 images to conserve memory
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
```
### Generator code:
```
###SETUP GENERATOR FUNCTION TO STREAM DATA INSTEAD OF PRE-LOADING INTO MEMORY
def generator(samples, batch_size=128):
  ....
  yield sklearn.utils.shuffle(inputs, outputs)
```
### Model Architecture 

#### 1. Solution Design Approach

I started with a generic dense network, which didnt work too well. I looked at the advanced Nvidia Architecture, but wanted to see if there was an easier fix, since I did not have access to the GPU. The Comma.AI steering model seemed like a good compromise between too simple and too complex of a network. 

The vehicle made it just fine through multiple rounds of track 1 with this model, with the training strategy mentioned above.

A video of the vehicle driving along the track can be found here in the file run1.mp4. The video is grainy, to save space I used 45 Frames per second. [run1.mp4](./run1.mp4)

