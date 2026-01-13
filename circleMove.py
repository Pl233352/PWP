'''
Version 1.1.2
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

#get video
video = cv.VideoCapture('IMG_1368.MOV')

if not video.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    #read a frame
    ret, frame = video.read()

    #if the frame was not read successfully, break the loop
    if not ret:
        print("End of video stream.")
        break

    #grayscale
    newvideo = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #blur
    newvideo=cv.medianBlur(newvideo, 11)

    #find circles
    circles = cv.HoughCircles(newvideo,cv.HOUGH_GRADIENT,dp=1,minDist=500,param1=120,param2=45,minRadius=260,maxRadius=270)

    #draw circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0]:
            if x==x:
                cv.circle(frame, (x, y), r, (0, 255, 0), 10)
                cv.circle(frame, (x, y), 2, (0, 0, 255), 3)
                print(x,y,r)

    #display the frame
    cv.imshow('Video Player', frame)

    #end video if x is pressed
    if cv.waitKey(25) & 0xFF == ord('x'):
        break

#end video
video.release()
cv.destroyAllWindows()

