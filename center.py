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
    
    #grayscale
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    
    #blur
    blur=cv2.medianBlur(gray, 25)

    #canny edge detection
    canny=cv2.Canny(blur,100,200)

    #apply mask
    #1280 960 video size
    '''
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height  = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    width = np.int32(width)
    height = np.int32(height)

    start_x=0
    start_y=height

    end_x=width
    end_y=height

    half_x=end_x//2
    half_y=end_y//2

    mask=np.zeros_like(frame)

    vertices = [
        (start_x, start_y),
        (half_x, half_y),
        (end_x, end_y),
    ]

    mask_color= (0,0,255)
    cv2.fillPoly(frame, vertices, mask_color)
    '''
    cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
    cv2.line(frame, (start_x, start_y), (half_x, half_y), (0, 0, 255), 2)
    cv2.line(frame, (end_x, end_y), (half_x, half_y), (0, 0, 255), 2)

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
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # (where, start_point, end_point, color, thickness)

    #display the frame
    cv2.imshow('Video Player', frame)

    #end video if x is pressed
    if cv2.waitKey(25) & 0xFF == ord('x'):
        break

#end video
video.release()
cv2.destroyAllWindows()


