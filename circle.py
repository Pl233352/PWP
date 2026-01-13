'''
Version 1.0.2
RK
--------------
1. Import libraries
2. Get image
3. Grayscale image
4. Blur image
5. Detect circles in the image
6. Draw circles
7. Draw center of circles
8. Show image

'''

#import libraries
import numpy as np
import cv2 as cv

#get image
img = cv.imread('IMG_1356.jpg')
#grayscale
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#blur
img=cv.medianBlur(img, 11)

#find circles
circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,dp=1,minDist=500,param1=120,param2=45,minRadius=500,maxRadius=1000)

#draw circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0]:
        if x==x:
            cv.circle(img, (x, y), r, (255, 255, 255), 10)
            cv.circle(img, (x, y), 2, (255, 255, 255), 3)
            print(x,y,r)

#show image
cv.imshow('detected circles',img)
cv.waitKey(0)
cv.destroyAllWindows()
