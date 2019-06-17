import numpy as np
import cv2
from matplotlib import pyplot as plt

# opening and saving as different file type
img = cv2.imread('image.png')
# img = cv2.imwrite('image.jpg', img)
# cv2.imshow('Original', img)

# drawing
pic = np.zeros((500, 500, 3), dtype='uint8')
# cv2.circle(pic, (250,250), 50, (255, 0, 255))
# cv2.line(pic, (0,0), (500,500), (0,255,0), 3, cv2.LINE_8)
# cv2.rectangle(pic, (250, 250), (350, 450), (230, 234, 128), 5, cv2.LINE_4)
cv2.putText(pic, 'Udemy', (100,100), cv2.FONT_HERSHEY_COMPLEX, 3, (255,255,255), 4, cv2.LINE_8)

# cv2.imshow('dark', pic)

# image manipulation - translating and rotating
pic = cv2.imread('image.png')
rows = pic.shape[0]
cols = pic.shape[1]
center = (cols/2, rows/2)
angle = 90

M = np.float32([[1, 0, 150], [0, 1, 70]])
Mrot = cv2.getRotationMatrix2D(center, angle, 1)  # last number is scale of image
# shifted = cv2.warpAffine(pic, M, (cols, rows))
rotated = cv2.warpAffine(pic, Mrot, (cols, rows))

# cv2.imshow('shifted', rotated)

# image thresholding
pic = cv2.imread('image.png', 0)
threshold_value = 200
(T_value,binary_threshold) = cv2.threshold(pic, threshold_value, 255, cv2.THRESH_BINARY)

# cv2.imshow('binary', binary_threshold)

# gaussian blur
pic = cv2.imread('balloons_noisy.png')
matrix = (7,7)
blur = cv2.GaussianBlur(pic, matrix, 0)

# cv2.imshow('Gaussian Blur', blur)

# median blur - for removing noise
pic = cv2.imread('balloons_noisy.png')
kernel = 3

median = cv2.medianBlur(pic, kernel)
# cv2.imshow('noisy', pic)
# cv2.imshow('median', median)


# bilateral filtering - for blurring while keeping edges, slower
pic = cv2.imread('pic.jpg')
d = 7  # diameter from center pixel to average
color = 100  # how many colors away from center pixel to be considered
space = 200 # how far from center pixel that pixels will be included

filtered = cv2.bilateralFilter(pic, d, color, space)
# cv2.imshow('orig', pic)
# cv2.imshow('filter', filtered)

#canny edge detector
pic = cv2.imread('image.png')
threshold_value1 = 50
threshold_value2 = 100

canny = cv2.Canny(pic, threshold_value1, threshold_value2)
# cv2.imshow('canny', canny)

cv2.waitKey(0)
cv2.destroyAllWindows()

#video player
vid = cv2.VideoCapture('Wildlife.mp4')
while (vid.isOpened()):
    ret, frame = vid.read()
    cv2.imshow('vid', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()

