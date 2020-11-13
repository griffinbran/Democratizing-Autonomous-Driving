# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Democratizing Autonomous Driving
---
### Problem Statement
The existing self-driving car market is dominated by multi-billion dollar companies such as Alphabet, Tesla, Ford etc. Perceived barriers to entry include high costs, access to data, and technicality. A market with only big companies will result in monopoly/oligopoly where a few companies get to decide the price of the product, not the supply and demand relation. Consumers suffer greatly from monopoly as they don't have other options. In an effort to prevent this situation from happening, we would like to build a self-driving car model from scratch based on convolutional neural networks to showcase that self-driving is not as beyond reach for smaller companies, or even startups. In fact, it is possible to build a well-performed self-driving car model with relative small datasets, simple hardware requirement and a straightforward process. By doing this, we hope that more company executives will join the wave of self-driving rather than sit back and be the observants.

  We will be exploring the following specific questions:
1. Can we train a model using convolutional neural network(CNN) to keep the car in the lane and finish a track?
2. Can we optimize the model to make the car drive more like a human (more smoothly), rather than swerve in the lane?
3. Can we optimize the model to drive as fast as it could while still stay in the lane?
4. Can we use a smaller dataset to achieve similar results as big datasets?
5. Can we break through the hardware limitations of smaller companies?

---

### Contents

* [Background](#background)
* [Data Aquisition & Cleaning](#data_aquisition_and_cleaning)
* [Exploratory Data Analysis](#exploratory_data_analysis)
* [Image Preprocessing & Augmentation](#image_preprocessing_&_augmentation)
* [Modeling and Tuning](#modeling_and_tuning)
* [Evaluation](#evaluation)
* [Findings and Recommendations](#findings_and_recommendations)
* [Limitations and Next Steps](#limitations_and_next_steps)
* [Technical Log](#technical_log)
* [Software Requirements](#software_requirements)
* [Acknowledgements and Contact](#acknowledgements_and_contact)


---
<a id='background'></a>
### Background

Here is some background info:
* Captured human steering angles are mapped to raw pixels, from timestamped video data, by CNNs to generate proposed steering commands
* End-to-End learning: Automated detection of useful road features, no explicit decomposition of processing pipeline such as path planning or control
* unpaved roads, without lane markings (or lane detection/guard rails/other cars), minimum data from humans
* No "human-selected intermediate criteria", system performance over human interpretation
* Training: Desired sterring commands = y_true, Proposed steering commands = y_pred, compute error --> Back propagated weight adjustment

---
<a id='data_aquisition_and_cleaning'></a>
### Data Aquisition & Cleaning

We luckily found a driving simulator open sourced by Udacity which allows us easily collect data. Once we hit record button, we can control the car with WASD keyboard and the simulator will automatically generate a driving log which records images captured by three cameras placed left, center and right at the car front and the WASD inputs. In order to validate our hypothesis that even with smaller datasets, we can still build a well-performed model with CNN, we fed into our model with two different datasets: Udacity-released datasets which has over 9,000 rows of data and a self-generated dataset with only 1300+ rows. 

Thanks to the auto-generated driving log, there are not much data cleaning for us to do except for adding the columns for data and align the file path for the camera captures.


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
|**Layer One**|*dtype*|Origin of Data|*Category*|*Description*|
|**variable2**|*dtype*|Origin of Data|*Category*|*Description*|
|**variable1**|*dtype*|Origin of Data|*Category*|*Description*|


|**CNN Architecture**|*Kernel Size*|*neurons*|No. of Images|*Stride*|*Shape ( h x w x RGB )*|
|---|---|---|---|---|---|
|**Input Layer**|*None*|None|< sample size >|None|*( 160 x 320 x 3 )*|
|**Convolution 01**|*( 5 x 5 )*|24|24|*( 2 x 2 )*|*(  78 x 158 x 3 )*|
|**Convolution 02**|*( 5 x 5 )*|36|864|*( 2 x 2 )*|*(  37 x  77 x 3 )*|
|**Convolution 03**|*( 5 x 5 )*|48|41,472|*( 2 x 2 )*|*(  16 x  36 x 3 )*|
|**Convolution 04**|*( 3 x 3 )*|64|2,654,208|*None*|*(  37 x  77 x 3 )*|
|**Convolution 05**|*( 3 x 3 )*|64|169,869,312|*None*|*(  16 x  36 x 3 )*|
|**Dropout**|*None*|None|169,869,312|*None*|*(  16 x  36 x 3 )*|
|**Flatten**|*None*|None|169,869,312|*None*|*(  16 x  36 x 3 )*|
|**Dense 01**|*None*|100|169,869,312|*None*|*(  16 x  36 x 3 )*|
|**Dense 02**|*None*|50|169,869,312|*None*|*(  16 x  36 x 3 )*|
|**Dense 03**|*None*|10|169,869,312|*None*|*(  16 x  36 x 3 )*|
|**Dense Output**|*None*|1|169,869,312|*None*|*(  16 x  36 x 3 )*|


|**CNN Model**|*Split*|*Epoch*|*Loss*|*Accuracy*|
|---|---|---|---|---|
|**Bseline MSE**|*Training*|01|0.0316|0.3251|
|**Bseline MSE**|*Validation*|01|0.0191|0.8220|
|**Bseline MSE**|*Training*|02|0.0266|0.3248|
|**Bseline MSE**|*Validation*|02|0.0205|0.8240|

|**CNN Model**|*Split*|*Epoch*|*Loss*|*Accuracy*|
|---|---|---|---|---|
|**Huber Loss, $\delta$=0.2**|*Training*|01|0.0243|0.3254|
|**Huber Loss, $\delta$=0.2**|*Validation*|01|0.0207|0.8245|
|**Huber Loss, $\delta$=0.2**|*Training*|02|0.0131|0.3247|
|**Huber Loss, $\delta$=0.2**|*Validation*|02|0.0097|0.8235|
|**Huber Loss, $\delta$=0.4**|*Training*|01|0.0158|0.3252|
|**Huber Loss, $\delta$=0.4**|*Validation*|01|0.0093|0.8227|
|**Huber Loss, $\delta$=0.4**|*Training*|02|0.0133|0.3249|
|**Huber Loss, $\delta$=0.4**|*Validation*|02|0.0103|0.8233|
|**Huber Loss, $\delta$=0.6**|*Training*|01|0.0160|0.3252|
|**Huber Loss, $\delta$=0.6**|*Validation*|01|0.0092|0.8225|
|**Huber Loss, $\delta$=0.6**|*Training*|02|0.0135|0.3249|
|**Huber Loss, $\delta$=0.6**|*Validation*|02|0.0103|0.8236|
|**Huber Loss, $\delta$=0.8**|*Training*|01|0.0160|0.3252|
|**Huber Loss, $\delta$=0.8**|*Validation*|01|0.0093|0.8213|
|**Huber Loss, $\delta$=0.8**|*Training*|02|0.0135|0.3249|
|**Huber Loss, $\delta$=0.8**|*Validation*|02|0.0099|0.8236|
|**Huber Loss, $\delta$=1.0**|*Training*|02|0.0134|0.3248|
|**Huber Loss, $\delta$=1.0**|*Validation*|02|0.0097|0.8235|

---
<a id='exploratory_data_analysis'></a>
### Exploratory Data Analysis

The EDA is based only on the Udacity-released dataset.

The mean of steering angles is 0, with standard deviation of 0.16. We could tell that the data does not have big spread, which makes sense as too much steering should not be expected on a gentle track. The distribution of steering angles shows that 0 steering angle is the abosolute most frequent number, which confirms that the track requires mostly straight driving. A breakdown of the counts of steering angles per direction indicate that there are more left turns than right turns, which means the dataset would be baised to favorleft turns than right turns. Also, as going straight is the aboslute majority, driving straight would also be preferred. To mitigate these issues, image augmentation should be counsidered to use on our datasets, for i.e.: we should consider filp the images to create a more balanced set between turns. And limit the number of zero steerings in the sample. 

<img src='./charts/dist_of_steering_angles.png' style="float: left; width: 375px;"/>                   
<img src='./charts/count_of_steering.png' style="float:right; width: 375px;"/>

By exploring the captured images from the three cameras, we found that the images from the center camera have in fact contained all the information that left and right cameras have captured. Therefore we believe the images from the center camera should be sufficient to be the only input. However, images captured from left and right could serve as our assitance in correcting the steering of the car in case it goes off center. So we would also include those images, only to adjust the steering angles accordingly with a correction angle. 

Furthermore, the original images contain irrelevant info such as sky, grass and car front which are noise rather than signal for the model. We therefore decided to crop those out of the images before feeding them into the convolutional neural networks.

![](./charts/neg_steer.png)
![](./charts/zero_steer.png)
![](./charts/pos_steer.png)

---
<a id='image_preprocessing_&_augmentation'></a>
### Image Preprocessing & Augmentation

To improve the quality of the data, image preprocessing and augmentation are applied. The preprocessing include cropping, resizing and converting the color space. Take cropping as an example, each image contains information irrelevant to the problem at hand, e.g. the sky, background, trees, hood of the car. The images are thus cropped to avoid feeding the neural network this superfluous information. Below is an example of how the images look after being cropped:

<img src='./charts/before_after/before.png' style="float: left; width: 500px;"/>                   
<img src='./charts/before_after/after_crop.png' style="float: right; width: 500px;"/>


During model training, images are also augmented in the way of random flip, translation, shadow, or brighten transformation. Augmentation of this sort creates a more robust model that is less overfit on the training set, as we are artificially introducing variance into the training.

Below are the examples of images to which random flip was applied:

<img src='./charts/before_after/after_flip.png' style="float: left; width: 375px;"/>                   

Examples of images to which random brightness was applied:

<img src='./charts/before_after/after_brightness.png' style="float: left; width: 1000px;"/>                   

Examples of images to which random shadow was applied:

<img src='./charts/before_after/after_shadow.png' style="float: left; width: 1000px;"/>  

---

<a id='modeling_and_tuning'></a>
### Modeling and Tuning

Our model is based on the NVIDIA model which explores the possibility of using only CNN regression.

Baseline Model

Best Model


---

<a id='evaluation'></a>
### Evaluation

We looked at MSE, Huber, and custom loss functions
MSE and Huber models make it around the track with a speed limit
Custom loss model drives more like a human but is still unable to make it around the track
All of the models cannot make it around the track without the speed limit. Albeit, the MSE and Huber models have significantly more unstable swerving than the custom loss model
Custom loss model stays on the track longer, and drives better with no speed limit than the other models.


---
<a id='findings_and_recommendations'></a>
### Findings and Recommendations

CNN is successful in keeping the car in the lane with a speed limit.
We were unable to construct a model that could complete a lap at full speed yet.
Custom loss function successful in changing the behavior of the car to be more human-like but unsuccessful in completing a lap with or without a speed limit.
A small amount of training data is still sufficient to train the model.
Hardware limitation can be resolved by using a batch generator.
CNN model is able to perform without lane marks


Mid-sized car companies: don’t be afraid to enter the market to compete!
Companies with long-haul, fixed routes who look to decrease the casualty of drivers due to fatigue driving/extreme road condition: this model could be very helpful.
Companies who cannot afford to transform their fleets: could opt for self-driving detection devices using this model.


---
<a id='limitations_and_next_steps'></a>
### Limitations and Next Steps

Limitations:

Due to the fact that our model is simply a regression model, the application of our model may be limited to predictable road conditions, such as driving on the highway in the midwest (interstate driving), rather than a traffic jam in LA (city driving).
Due to the time limitations, we were not able to run too many epochs to test out if the model can generalize to other tracks. Even with GPU cloud computing, training times are long, ~40 min per epoch.

Next Steps:

1. Increase the epochs of training to evaluate the model performance on a different track
2. Further augment the images to include different weather conditions
3. Utilize transfer learning and experiment with more complex road conditions (city driving, change lanes, parking, etc)

---
<a id='technical_log'></a>
### Technical Log:
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

> * 10/29/2020 Discuss training ...
> * and then..
> * and then...

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

Contact:

> * Anthony Clemens ([GitHub](https://git.generalassemb.ly/ajclemens) | [LinkedIn](https://www.linkedin.com/in/anthony-clemens/))
> * Cloudy Liu      ([GitHub](https://git.generalassemb.ly/cloudmcloudyo) | [LinkedIn](https://www.linkedin.com/in/cloudyliu/))
> * Brandon Griffin ([GitHub](https://github.com/griffinbran) | [LinkedIn](https://www.linkedin.com/in/griffinbran/))

Project Link: ([*source*](https://git.generalassemb.ly/cloudmcloudyo/optimizing-self-driving/blob/master/README.md))