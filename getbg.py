import cv2
import numpy as np


nofog = cv2.imread(r'E:\Develop\python\League-X\Mini Figures\nofog.jpg')
bg = cv2.imread(r'E:\Develop\python\League-X\Mini Figures\bg.jpg')
emptybase = cv2.imread(r'E:\Develop\python\League-X\Mini Figures\emptybase.jpg')
#bg[1317:1423, 2190:2298] = nofog[1317:1423, 2190:2298]
bg[1068:1137, 2470:2540] = emptybase[1068:1137, 2470:2540]


cv2.imshow('test', bg)
cv2.waitKey()
#cv2.imwrite(r'E:\Develop\python\League-X\Mini Figures\bg.jpg', bg)