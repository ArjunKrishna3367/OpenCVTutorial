
from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from PIL import Image

import scipy.misc

import PIL
from PIL import Image
im = Image.open("dataCollection/person_images/img133790.jpg")
width, height = im.size
im = im.resize((120, int(120/width*height)), PIL.Image.NEAREST)
im.save("result.png")

# image = plt.imread("result.png")
# gray = rgb2gray(image)
# print(gray.shape[0] * gray.shape[1])
#
# gray_r = gray.reshape(gray.shape[0] * gray.shape[1])
# for i in range(gray_r.shape[0]):
#     print(i)
#     if gray_r[i] > gray_r.mean():
#         gray_r[i] = 1
#     else:
#         gray_r[i] = 0
# gray = gray_r.reshape(gray.shape[0], gray.shape[1])
# plt.imshow(gray, cmap='gray')
#
# gray = rgb2gray(image)
# gray_r = gray.reshape(gray.shape[0] * gray.shape[1])
# print(gray_r.mean())
# for i in range(gray_r.shape[0]):
#     if gray_r[i] > gray_r.mean():
#         gray_r[i] = 3
#     elif gray_r[i] > 0.5:
#         gray_r[i] = 2
#     elif gray_r[i] > 0.25:
#         gray_r[i] = 1
#     else:
#         gray_r[i] = 0
# gray = gray_r.reshape(gray.shape[0], gray.shape[1])
# plt.imshow(gray, cmap='gray')
#
# # defining the sobel filters
# sobel_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])])
#
# sobel_vertical = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])
#
# out_h = ndimage.convolve(gray, sobel_horizontal, mode='reflect')
# out_v = ndimage.convolve(gray, sobel_vertical, mode='reflect')
# # here mode determines how the input array is extended when the filter overlaps a border.
# plt.imshow(out_h, cmap='gray')
# plt.imshow(out_v, cmap='gray')
#
# kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
#
# out_l = ndimage.convolve(gray, kernel_laplace, mode='reflect')
# plt.imshow(out_l, cmap='gray')

pic = plt.imread('result.png') # dividing by 255 to bring the pixel values between 0 and 1
print(pic.shape)
plt.imshow(pic)
plt.show()

pic_n = pic.reshape(pic.shape[0] * pic.shape[1], pic.shape[2])

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]


cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
plt.imshow(cluster_pic)
plt.show()
