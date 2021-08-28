import cv2
import numpy as np
import time
import os
import tensorflow as tf
import time
import class_champions

# rescale values
def rescale(img, orig, new):
    img = np.divide(img, orig)
    img = np.multiply(img, new)
    img = img.astype(np.uint8)
    return img

# get abs(diff) of all hue values
def diff(bg, fg):
    # do both sides
    lh = bg - fg
    rh = fg - bg

    # pick minimum # this works because of uint wrapping
    low = np.minimum(lh, rh)
    return low

def handle(bg, fg):
    fg_original = fg.copy()


    # convert to lab
    bg_lab = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB)
    fg_lab = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB)
    bl, ba, bb = cv2.split(bg_lab)
    fl, fa, fb = cv2.split(fg_lab)

    # subtract
    d_b = diff(bb, fb)
    d_a = diff(ba, fa)

    # rescale for contrast
    d_b = rescale(d_b, np.max(d_b), 255)
    d_a = rescale(d_a, np.max(d_a), 255)

    # combine
    combined = np.maximum(d_b, d_a)

    # threshold 
    # check your threshold range, this will work for
    # this image, but may not work for others
    # in general: having a strong contrast with the wall makes this easier
    thresh = cv2.inRange(combined, 80, 255)
    # opening and closing
    #kernel = np.ones((5,5), np.uint8)
    kernelSizes = [(3, 3), (5, 5), (7, 7)]
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))

    # closing
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, 2)

    # opening
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, 2)

    # contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    # filter contours by size
    big_cntrs = []
    rects = []
    marked = fg_original.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 120:
            big_cntrs.append(contour)
            rects.append(cv2.boundingRect(contour))
    cv2.drawContours(marked, big_cntrs, -1, (0, 255, 0), 1)

    # create a mask of the contoured image
    mask = np.zeros_like(fb)
    mask = cv2.drawContours(mask, big_cntrs, -1, 255, -1)
    #boundRect = cv2.boundingRect(big_cntrs)
    #mask = cv2.rectangle(mask, int(boundRect[0]), int(boundRect[1]), (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (0, 255, 0), 2)

    # erode mask slightly (boundary pixels on wall get color shifted)
    #mask = cv2.erode(mask, kernel, iterations = 1)

    # crop out
    out = np.zeros_like(fg_original) # Extract out the object and place into output image
    #out[mask == 255] = fg_original[mask == 255]
    for boundRect in rects:
        x = boundRect[0]
        y = boundRect[1]
        w = boundRect[2]
        h = boundRect[3]
        out[y:y+h,x:x+w] = fg_original[y:y+h,x:x+w]
    return out

image_path = r'E:\Develop\python\League-X\Mini Figures\test'
kernel = np.ones((1,1),np.uint8)

bg = cv2.imread(r'E:\Develop\python\League-X\Mini Figures\bg.jpg')
bg = bg[1060:1800, 2190:2550]
#bg = cv2.blur(bg,(5,5))
#bg = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
##bgb,bgg,bgr,bga = cv2.split(bg)
#bgr = cv2.inRange(bgr,120,255)
#bgg = cv2.inRange(bgg,120,255)
#bgb = cv2.inRange(bgb,120,255)
#ind0 = bgb - bgr - bgg

#backSub = cv2.createBackgroundSubtractorKNN()

for name in os.listdir(image_path):
    file = os.path.join(image_path, name)
    img = cv2.imread(file)
    img = img[1060:1800, 2190:2550]
    #img = cv2.blur(img,(5,5))

    crop = handle(bg, img)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    circles = None
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=15, minRadius=10, maxRadius=30)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(len(circles[0, :]))
        for i in circles[0, :]:
            center = (i[0], i[1])
            x = i[0]
            y = i[1]
            radius = i[2]

            cv2.circle(img, center, radius+10, (255, 255, 0), 3)

    #cv2.imshow('crop', blank_image)
    
    cv2.imshow('crop', crop)
    cv2.imshow('imagea', img)
    cv2.imshow('imageb', gray)
    # setting mouse hadler for the image
    # and calling the click_event() function
    #cv2.setMouseCallback('image', move_event)

    cv2.waitKey(0)

cv2.destroyAllWindows()