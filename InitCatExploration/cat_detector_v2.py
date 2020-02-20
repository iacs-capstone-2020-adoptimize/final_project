import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cat_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
cat_cascade_ext = cv2.CascadeClassifier("haarcascade_frontalcatface_extended.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
SF = 1.3
N = 3
def processImage(image_dir, image_filename):
    img = cv2.imread(image_dir + image_filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cats = cat_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors = N)
    cats_ext = cat_cascade_ext.detectMultiScale(gray, scaleFactor=SF, minNeighbors = N)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors = N)
    print(eyes)
    print(cats)

    for (x, y, w, h) in cats:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    for (x, y, w, h) in cats_ext:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x, y, w, h) in eyes:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imwrite("images/out"+image_filename, img)

processImage("images/", "cat-04.jpg")
