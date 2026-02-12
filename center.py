'''
Version 1.1.2
RK
--------------
1. Import libraries
2. Get video stream
3. Grayscale video stream
4. Blur video stream
5. Apply Canny edge detection on video stream
6. Crop video stream (apply mask)
7. Detect lines on video stream
8. Draw lines of video stream
9. Display video stream
10. Terminate video stream when 'x' is pressed.
'''

#import libraries
import numpy as np
import cv2 as cv2

#get video
video = cv2.VideoCapture(0)

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

    #HSV
    #nothing here yet
    
    #grayscale
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    
    #blur
    blur=cv2.medianBlur(gray, 25)
    
    #canny edge detection
    canny=cv2.Canny(blur,100,200)
    
    #apply mask
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height  = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    width = int(width)
    height = int(height)
    
    start_x=width//4
    start_y=height//4
    
    end_x=start_x*3
    end_y=start_y*3
    
    cropped = canny[start_y:end_y, start_x:end_x]
    cv2.line(frame, (start_x, start_y), (end_x, start_y), (0, 0, 255), 2)
    cv2.line(frame, (start_x, start_y), (start_x, end_y), (0, 0, 255), 2)
    cv2.line(frame, (end_x, end_y), (end_x, start_y), (0, 0, 255), 2)
    cv2.line(frame, (end_x, end_y), (start_x, end_y), (0, 0, 255), 2)
    
    #houghlines
    lines=cv2.HoughLinesP(cropped, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=100)
    if lines is not None:
        #lineList=[]
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1+start_x, y1+start_y), (x2+start_x, y2+start_y), (0, 255, 0),2)
            #lineList.append(line[0][0])
            #lineList.append(line[0][1])
            #lineList.append(line[0][2])
            #lineList.append(line[0][3])
    #abcd
    
    '''
    #circles
    circles = cv2.HoughCircles(cropped,cv2.HOUGH_GRADIENT,dp=1,minDist=500,param1=120,param2=45,minRadius=260,maxRadius=270)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0]:
            if x==x:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 10)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                print(x,y,r)
    '''
    
    #display the original frame
    cv2.imshow('', frame)
    #end video if x is pressed
    if cv2.waitKey(25) & 0xFF == ord('x'):
        break
        
#end video
video.release()
cv2.destroyAllWindows()

#end of code :D 




