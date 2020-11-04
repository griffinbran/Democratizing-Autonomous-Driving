# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Optimizing Autonomous Driving
---
### Problem Statement
***TO DO***

  Questions to be explored:
> 1. Point 1...
> 2. Point 2...
> 3. What is a reasonable 'optical cue'?
<br>
---
### Overview

This DSI module covers:

- Machine Learning for Deep Neural Networks (TensorFlow, Keras API)
- Cloud Coumputing with a GPU (Google Colaboratory, CNN implementation with parallel processing)
- Computer Vision ( RGB image processing, image formation, feature detection, computational photography)
- Convolutional Neural Networks (regularization, automated pattern recognition, )
- Intelligent autonomous systems (self-driving cars)

### Contents

* [Background](#background)
* [Data Aquisition & Cleaning](#data_aquisition_and_cleaning)
* [Exploratory Analysis](#exploratory_analysis)
* [Findings and Recommendations](#findings_and_recommendations)
* [Next Steps](#next_steps)
* [Software Requirements](#software_requirements)
* [Acknowledgements and Contact](#acknowledgements_and_contact)
---
<a id='background'></a>
### Background

Here is some background info:
> * Captured human steering angles are mapped to raw pixels, from timestamped video data, by CNNs to generate proposed steering commands
> * End-to-End learning: Automated detection of useful road features, no explicit decomposition of processing pipeline such as path planning or control
> * unpaved roads, without lane markings (or lane detection/guard rails/other cars), minimum data from humans
> * No "human-selected intermediate criteria", system performance over human interpretation
> * Training: Desired sterring commands = y_true, Proposed steering commands = y_pred, compute error --> Back propagated weight adjustment

### Data Dictionary

**NOTE: Make sure you cross-reference your data with your data sources to eliminate any data collection or data entry issues.**<br>
*See [Acknowledgements and Contact](#acknowledgements_and_contact) section for starter code resources*<br>

|Feature|Type|Dataset|Category|Description|
|---|---|---|---|---|
|**variable1**|*dtype*|Origin of Data|*Category*|*Description*|
|**variable2**|*dtype*|Origin of Data|*Category*|*Description*|
|**IMAGE_HEIGHT**|*int*|utils.py|*Global Variable*|*160(pixels)-Vertical units across: Top=0 to Bottom= 159*|
|**IMAGE_WIDTH**|*int*|utils.py|*Global Variable*|*320(pixels)-Horizontal units across: Left=0 to Right= 319*|
|**IMAGE_CHANNELS**|*int*|utils.py|*Global Variable*|*3-RGB Channels*|
|**INPUT_SHAPE**|*3-tuple*|utils.py|*Global Variable*|*(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)*|

---
<a id='data_aquisition_and_cleaning'></a>
### Data Aquisition & Cleaning
#### Cloning and Debugging

> * 10/27/2020 Pre-trained simulator is downloaded and run. First data collection.
> * 10/28/2020 An updated version of Keras and a dated starter code lead to a rocky start full of error messages.
> * 10/28/2020 After no success working in a virtual enviornment, Anthony brute force debugs *model.py* and *utils.py* from outdated dependencies.
> * and then...

#### Cloud Computing with GPU

> * 10/28/2020 Anthony attempted running *model.py* on his local machine and reported 2-3 hours per epoch. The workload clearly needs to be shifted for reduced computation time as well as for practical reasons.
> * 10/28/2020 Cloudy successfully imports *utils.py* to the cloud --> from utils import INPUT_SHAPE, batch_generator, and we successfully upload *model.py* in Google Colaboratory. 
> * 10/28/2020 Anthony runs 3 epochs overnight.
> * 10/29/2020 Success! Three epochs were enough to start our self-driving car...a swerving and shifty self-driving car, but one none-the-less.

#### Training the CNN

> * Network architecture: 9 layers, 5 convolution layers, 3 fully connected layers
> * and then..
> * 

def build_model(args):
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)

    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model

---
<a id='exploratory_analysis'></a>
### Exploratory Analysis

> * Insert EDA details...
> *
> *

**Data Cleaning and EDA**
- Does the student fix data entry issues?
- Are data appropriately labeled?
- Are data appropriately typed?
- Are datasets combined correctly?
- Are appropriate summary statistics provided?
- Are steps taken during data cleaning and EDA framed appropriately?

### Data Visualization

> * Make some pretty plots with Tableau:

**Visualizations**
- Are the requested visualizations provided?
- Do plots accurately demonstrate valid relationships?
- Are plots labeled properly?
- Plots interpreted appropriately?
- Are plots formatted and scaled appropriately for inclusion in a notebook-based technical report?

---
<a id='findings_and_recommendations'></a>
### Findings and Recommendations

  Answer the problem statement:
> 1. Point 1...
> 2. Point 2...
> 3. Point 3...

---
<a id='next_steps'></a>
### Next Steps:

---
<a id='software_requirements'></a>
### Software Requirements:


---
<a id='acknowledgements_and_contact'></a>
### Acknowledgements and Contact:

External Resources:
* [`How to Simulate a Self-Driving Car`] (YouTube): ([*source*](https://www.youtube.com/watch?v=EaY5QiZwSP4&t=1209s))
* [`udacity/self-driving-car-sim`] (GitHub): ([*source*](https://github.com/udacity/self-driving-car-sim))
* [`naokishibuya/car-behavioral-cloning`] (GitHub): ([*source*](https://github.com/naokishibuya/car-behavioral-cloning))
* [`llSourcell/How_to_simulate_a_self_driving_car`] (GitHub): ([*source*](https://github.com/llSourcell/How_to_simulate_a_self_driving_car))

Papers:
* `End-to-End Deep Learning for Self-Driving Cars` (NVIDIA Developer Blog): ([*source*](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/))
* `Explaining How End-to-End Deep Learning Steers a Self-Driving Car` (NVIDIA Developer Blog): ([*source*](https://developer.nvidia.com/blog/explaining-deep-learning-self-driving-car/))
* `End to End Learning for Self-Driving Cars` (arXiv): ([*source*](https://arxiv.org/pdf/1604.07316v1.pdf))
* `VisualBackProp: efficient visualization of CNNs` (arXiv): ([*source*](https://arxiv.org/pdf/1611.05418.pdf))

### Contact:

> * Anthony Clemens ([GitHub](https://git.generalassemb.ly/ajclemens) | [LinkedIn](https://www.linkedin.com/in/anthony-clemens/))
> * Cloudy Liu      ([GitHub](https://git.generalassemb.ly/cloudmcloudyo) | [LinkedIn](https://www.linkedin.com/in/cloudyliu/))
> * Brandon Griffin ([GitHub](https://github.com/griffinbran) | [LinkedIn](https://www.linkedin.com/in/griffinbran/))

Project Link: ([*source*](https://git.generalassemb.ly/cloudmcloudyo/optimizing-self-driving/blob/master/README.md))

---
### Submission

**Materials must be submitted by 4:59 PST on Friday, November 13, 2020.**

---