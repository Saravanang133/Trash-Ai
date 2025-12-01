import cv2
import os
import string
import random
from os import listdir
from os.path import isfile, join, splitext
import time
import sys
import numpy as np
import argparse

watch_folder = 'toprocess'
processed_folder = 'processed'
poll_time = 1

before = dict([(f, None) for f in os.listdir(watch_folder)])
while 1:
    time.sleep(poll_time)
    after = dict([(f, None) for f in os.listdir(watch_folder)])
    added = [f for f in after if not f in before]
    removed = [f for f in before if not f in after]
    if added:
        print("Added ", ", ".join(added))
    if added[0] is not None:
        processImage(added[0])
    if removed:
        print("Removed ", ", ".join(removed))
    before = after

def processImage(fileName):
    image = cv2.imread(watch_folder + '/' + fileName)
    output = image
    hMin = 29  # Hue minimum
    sMin = 30  # Saturation minimum
    vMin = 0   # Value minimum (Also referred to as brightness)
    hMax = 179 # Hue maximum
    sMax = 255 # Saturation maximum
    vMax = 255 # Value maximum
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # Converting color space from BGR to HSV
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    *_, alpha = cv2.split(output)
    dst = cv2.merge((output, alpha))
    output = dst
    dim = (512, 512)
    output = cv2.resize(output, dim)
    file_name = randomString(5) + '.png'
    cv2.imwrite(processed_folder + '/' + file_name, output)

def randomString(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))










    

