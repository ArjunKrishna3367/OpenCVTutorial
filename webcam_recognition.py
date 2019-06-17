import cv2
import numpy as np

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

videocapture = cv2.VideoCapture(0)
scale_factor = 1.3

while 1:
    ret, pic = videocapture.read()

    faces = cascade.detectMultiScale(pic, scale_factor, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(pic, (x, y), (x + w, y + h), (255, 0, 0), 2)

    print("Number of faces found {} ".format(len(faces)))
    cv2.imshow('face', pic)
    k = cv2.waitKey(10) & 0xff
    if k == ord('q'):
        break
cv2.destroyAllWindows()