from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
import sys
import argparse
from math import atan2, cos, sin, sqrt, pi
#################### this function draw the axis ###################"
def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

######## this function calculates the Principal Component Analysis PCA for group of points in our case the points are contours ###########     
def getOrientation(pts, img):
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean) # opencv2 function that calculate PCA we get the eigen values and  eigenvector  
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    ######### plot the center of the objects with a circle    
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) ##### calculate the orientation in radians
    
    return angle

# Read the image and transform it to HSV color space
img_path = sys.argv[1]
test = sys.argv[2]
img = cv2.imread(img_path)
print(test)
if test == 'Noise':
   print(test)
   img = cv2.bilateralFilter(img, 11, 17, cv2.BORDER_ISOLATED)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('origin',img)
# Threshold with inRange() get only specific colors
lower_table_color = np.array([0,0,0])
upper_table_color = np.array([20,180,250])
mask = cv2.inRange(hsv, lower_table_color, upper_table_color)


# 
cv2.imshow('color segmentation', mask)
kernel = np.ones((28,28),np.uint8)
dilatation = cv2.dilate(mask,kernel,iterations = 1)
cv2.imshow('after dilatation',dilatation)
kernel = np.ones((19,19	),np.uint8)
erosion = cv2.erode(dilatation,kernel,iterations = 1)
cv2.imshow('after erosion', erosion)


top = int(0.05 * mask.shape[0])  # shape[0] = rows
bottom = top
left = int(0.05 * mask.shape[1])  # shape[1] = cols
right = left
value = [255,255,255]
borderType = cv2.BORDER_CONSTANT #add white border to the image to separate the the object from the original borders
mask = cv2.copyMakeBorder(erosion, top, bottom, left, right, borderType, None, value)
src = cv2.copyMakeBorder(img, top, bottom, left, right, borderType, None, value)

contours, hier= cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # calculate contours
cnt = sorted(contours, key=cv2.contourArea, reverse=True)
# cv2.drawContours(mask, cnt[1],-1,(0,255,0),2)
# cv2.drawContours(mask, cnt[2],-1,(0,255,0),2)
angl1 = getOrientation(cnt[2], src)*18000/314 #transform radian to degree
angl2 = getOrientation(cnt[1], src)*18000/314 #transform radian to degree
# print("angle order:",angl1,angl2)
angle0 = abs(angl1-angl2)
if angle0 >=180:
   angle0 = 180 - angle0
print('angle sorted', abs(angl1-angl2))
angl = abs(angl1 - angl2)
cv2.putText(src,str(angle0),(100,100),cv2.FONT_HERSHEY_SIMPLEX,3,(200,120,0),2,cv2.LINE_AA)

# Display results
cv2.imshow('img ', src)
cv2.waitKey(0)
cv2.destroyAllWindows()
