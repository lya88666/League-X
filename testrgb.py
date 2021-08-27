import cv2
import numpy as np
import time
import os
import tensorflow as tf
import time
import class_champions
from PIL import ImageGrab

def move_event(event, x, y, flags, params):
    imgk = img.copy()
    # checking for right mouse clicks     
    if event==cv2.EVENT_MOUSEMOVE:
  
        # displaying the coordinates
        # on the Shell
        # print(x, ' ', y)
  
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (x, y)
        B = imgk[y, x, 0]
        G = imgk[y, x, 1]
        R = imgk[y, x, 2]

    print('(x, y)=({}, {})'.format(x, y))

image_path = r'E:\Develop\python\League-X\Mini Figures\test'
kernel = np.ones((1,1),np.uint8)

for name in os.listdir(image_path):
    file = os.path.join(image_path, name)
    img = cv2.imread(file)
    img = img[1060:1800, 2190:2550]

    #cv2.waitKey()
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    b,g,r,a = cv2.split(dst)
    inranger = cv2.inRange(r,150,255)
    inrangeg = cv2.inRange(g,150,255)
    inrangeb = cv2.inRange(b,150,255)

    induction = inranger - inrangeg - inrangeb

    #induction = cv2.erode(induction, kernel,iterations = 1)
    #induction = cv2.medianBlur(induction, 1)
    rows = induction.shape[0]

    circles = cv2.HoughCircles(induction, cv2.HOUGH_GRADIENT, 1, 10,
                                    param1=100, param2=15, minRadius=5, maxRadius=20)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(len(circles[0, :]))
        for i in circles[0, :]:
            center = (i[0], i[1])

            # circle outline
            radius = i[2]
            cv2.circle(img, center, radius+10, (255, 255, 0), 3)

    cv2.imshow('crop', img)
    #cv2.imshow('imageb', induction)
    # setting mouse hadler for the image
    # and calling the click_event() function
    #cv2.setMouseCallback('image', move_event)

    cv2.waitKey(0)

cv2.destroyAllWindows()