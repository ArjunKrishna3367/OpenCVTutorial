# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import json
from tensorflow import keras
import tensorflow as tf


np.set_printoptions(threshold=np.inf)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())

with open("export-2019-07-02T14-20-03.918Z.json", "r") as read_file:
	data = json.load(read_file)

# load the image and apply SLIC and extract (approximately)
# the supplied number of segments
def readImage(name):
	image = cv2.imread("dataCollection/person_images/" + name)
	segments = slic(img_as_float(image), n_segments = 100, sigma = 5)
	return image, segments

def plot():
	# show the output of SLIC
	fig = plt.figure("Superpixels")
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
	plt.axis("off")
	plt.show()


def showSuperPixel(num):
	mask = np.zeros(image.shape[:2], dtype="uint8")
	mask[segments == num] = 255

	# show the masked region
	# cv2.imshow("Mask", mask)
	cv2.imshow("Applied", cv2.bitwise_and(image, image, mask=mask))
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def getSuperPixelLoc(num):
	pixelGroup = []
	# true false 2d array where true means that pixel is in superpixel num
	# each row of array = 1 row of picture
	equalsArr = (segments == num)
	equalsArr = np.array(equalsArr)
	print(equalsArr.shape)
	print(equalsArr)
	indexes = np.where(equalsArr==True)
	return indexes
	# loop through each row of picture
	# for (j, arrVals) in enumerate(equalsArr):
	# 	if True in arrVals:
	# 		for (k, arrVal) in enumerate(arrVals):
	# 			# if value is true (pixel is in group num)
	# 			if (arrVal):
	# 				# print RGB values of pixel
	# 				pixelGroup.append((j, k))
	# return pixelGroup


def getColors(num):
	pixelGroup = getSuperPixelLoc(num)

	def avg(list):
		return sum(list) / len(list)

	group4R = []
	group4B = []
	group4G = []
	for (j, k) in pixelGroup:
		group4B.append(image[j][k][0])
		group4G.append(image[j][k][1])
		group4R.append(image[j][k][2])
	return [avg(group4R), avg(group4G), avg(group4B)]


train_x = []
train_y = []
imageData = data[0]


# for imageData in data:
# image, segments = readImage(imageData['External ID'])
# print(imageData['External ID'])
# for (i, segVal) in enumerate(np.unique(segments)):
# 	train_x.append(getSuperPixelLoc(segVal))
# print(train_x[0][0])
# train_y.append(imageData['Label']['cap_or_no_cap'])
#
# print (train_x)
# image, segments = readImage('img68221.jpg')
# plot()


# loop over the unique segment values
# for (i, segVal) in enumerate(np.unique(segments)):
# 	print(segVal)
# 	print(getColors(segVal))
# 	showSuperPixel(segVal)

# model = keras.Sequential()
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(keras.layers.Dropout(rate=0.25, noise_shape=None, seed=None))
# model.add(keras.layers.Dense(64, activation=tf.nn.relu))
# model.add(keras.layers.Dense(64, activation=tf.nn.relu))
# model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
#
# model.compile(optimizer='adamax',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(allImages, train_y, epochs=10, shuffle=True)
#
# predictions = model.predict(testImages)

# def plot_image(i, predictions_array, true_label, img):
#     predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(img, cmap=plt.cm.binary)
#
#     predicted_label = np.argmax(predictions_array)
#     if predicted_label == true_label:
#         color = 'blue'
#     else:
#         color = 'red'
#
#     plt.xlabel("{} {:2.0f}% ({})".format(classNames[predicted_label],
#                                          100 * np.max(predictions_array),
#                                          classNames[true_label]),
#                color=color)
#
#
# # Method to plot bar chart of prediction probablities
# def plot_value_array(i, predictions_array, true_label):
#     predictions_array, true_label = predictions_array[i], true_label[i]
#     plt.grid(False)
#     plt.xticks(range(10), classNames, rotation=90)
#     plt.yticks([])
#     thisplot = plt.bar(range(10), predictions_array, color="#777777")
#     plt.ylim([0, 1])
#     predicted_label = np.argmax(predictions_array)
#
#     thisplot[predicted_label].set_color('red')
#     thisplot[true_label].set_color('blue')
#
#
# '''Plot image and chart of first image predictions'''
# for i in range(20):
#     plt.figure(figsize=(6,3))
#     plt.subplot(1,2,1)
#     plot_image(i, predictions, testLabels, testImages)
#     plt.subplot(1,2,2)
#     plot_value_array(i, predictions,  testLabels)
# plt.show()
