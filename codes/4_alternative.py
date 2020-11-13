import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2, os
import tensorflow as tf
from keras.models import load_model

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from platform import python_version
print(python_version())
import sys
#print(sys.version)
print(tf.__version__)

# Visualize accuracy scores of model after trained on data
def plot_loss(history, model_name):
    train_loss = history['loss'] 
    test_loss = history['val_loss'] 
    epoch_labels = list(range(1,16))
    # Set figure size
    plt.figure(figsize=(12, 8)) 
    # Generate line plot of training, testing loss over epochs
    plt.plot(train_loss, label='Training Loss', color='darkorchid', linestyle = '--') 
    plt.plot(test_loss, label='Validation Loss', color='firebrick') 
    # Set title
    plt.title(f'Loss by Epoch', fontsize=25)
    plt.xlabel('Epoch', fontsize=18) 
    plt.ylabel(r'Huber Loss, $\delta$ = 1.0', fontsize=18)
    plt.xticks(epoch_labels, epoch_labels)
    plt.legend(fontsize=18);

history_model_huber_15 = {"loss": [0.007388241123408079, 0.005166510120034218, 0.0044770450331270695, 0.0040555777959525585, 0.0037948722019791603,
                                  0.0035333053674548864, 0.0033559747971594334, 0.003178239334374666, 0.00305375549942255, 0.0029158128891140223,
                                  0.0028007295913994312, 0.0027071554213762283, 0.0026188171468675137, 0.0025329170748591423, 0.002458816161379218],
                         "val_loss": [0.009547244757413864, 0.009313859045505524, 0.009310279041528702, 0.009362826123833656, 0.009203005582094193,
                                      0.009230180643498898, 0.009388945065438747, 0.009333822876214981, 0.009061269462108612, 0.010325492359697819,
                                      0.009470313787460327, 0.009834242053329945, 0.009628728032112122, 0.009446615353226662, 0.009644902311265469]}
plot_loss(history_model_huber_15, 'Huber')
#plt.savefig('../charts/huber_loss_per_epoch.png')
#plt.show()

# Load model
from keras.models import load_model
model_path = './models/'
loaded_model_huber = load_model(model_path+'model-huber.h5')

loaded_model_huber.summary()

# summarize filter shapes
for layer in loaded_model_huber.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer, layer.name, filters.shape)

# summarize feature map shapes
for i in range(len(loaded_model_huber.layers)):
	layer = loaded_model_huber.layers[i]
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# summarize output shape
	print(i, layer.name, layer.output.shape)

# Retrieve weights from FIRST Conv2D Layer
filters1, biases1 = loaded_model_huber.layers[1].get_weights()
print('=====================================================')
print('Conv2D Layer 01:')
# (5 x 5) x 3-depth x 24-Activation Maps
print(filters1.shape)

# Retrieve weights from SECOND Conv2D Layer
filters2, biases2 = loaded_model_huber.layers[2].get_weights()
print('=====================================================')
print('Conv2D Layer 02:')
# (5 x 5)-kernel_size x 24-depth x 36-Activation Maps
print(filters2.shape)

# Retrieve weights from THIRD Conv2D Layer
filters3, biases3 = loaded_model_huber.layers[3].get_weights()
print('=====================================================')
print('Conv2D Layer 03:')
# (5 x 5)-kernel_size x 36-depth x 48-Activation Maps
print(filters3.shape)

# Retrieve weights from FOURTH Conv2D Layer
filters4, biases4 = loaded_model_huber.layers[4].get_weights()
print('=====================================================')
print('Conv2D Layer 04:')
# (3 x 3)-kernel_size x 48-depth x 64-Activation Maps
print(filters4.shape)

# Retrieve weights from FIFTH Conv2D Layer
filters5, biases5 = loaded_model_huber.layers[5].get_weights()
print('=====================================================')
print('Conv2D Layer 05:')
# (3 x 3)-kernel_size x 64-depth x 64-Activation Maps
print(filters5.shape)

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)


# plot first few filters
n_filters, ix = 24, 1
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    # plot each channel separately
    for j in range(3):
        # specify subplot and turn of axis
        ax = plt.subplot(n_filters, 3, ix, fc='grey')
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            c = '\nRED'
        elif j == 1:
            c = '\nGREEN'
        else:
            c ='\nBLUE'
        plt.xlabel(xlabel=c,color='w')
        plt.ylabel(ylabel=f'  \n\n{i+1}',color='w',rotation=0)
        # plot filter channel in grayscale
        plt.imshow(f[:, :, j], cmap='gray')
        ix += 1
print('\t\t FILTER #')
#plt.show()

# Output: Activation following 1st Conv2D layer
layer_2 = tf.keras.Model(inputs=loaded_model_huber.inputs, outputs=loaded_model_huber.layers[5].output)

# load the image with the required shape & add channels (3-RGB)
img_path = '../../../../Desktop/dsi/projects/IMG/center_2020_11_08_19_20_22_873.jpg'

# convert the image to an array
img = img_to_array(load_img(img_path, target_size=(160, 320)))

# expand dimensions so that it represents a single 'sample'
print(img.shape)
img = np.expand_dims(img, axis=0)
print(img.shape)

# get feature map for first hidden layer
feature_maps = layer_2.predict(img)


# plot all 64 maps in an 8x8 squares
row = 8
col = 8
square = 6
ix = 1
for _ in range(row):
	for _ in range(col):
		# specify subplot and turn of axis
		ax = plt.subplot(row, col, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
plt.show()
