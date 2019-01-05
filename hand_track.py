import cv2
import numpy as np
import math
from collections import deque

#painting stuff inspired by : #https://github.com/akshaychandra21/Webcam_Paint_OpenCV

'''
To Do:
- make colours a dictionary
'''

'''
Program that does the following:
- thresholds the hand
- draws contour and convex hull around the hand
- draws purple dot in the centre of hand
- draws light blue at fingertip
- tracks the fingertip and paints where it goes
'''

cap = cv2.VideoCapture(0)
cv2.namedWindow('output',cv2.WINDOW_NORMAL)
cv2.resizeWindow('output', 600,600) #resizing the window

# Setup deques to store separate colors in separate arrays
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

bindex = 0
gindex = 0
rindex = 0
yindex = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0
user_currently_painting = False

# Setup the Paint interface
paintWindow = np.zeros((471,636,3)) + 255 #the white canvas for the paint application
paintWindow = cv2.rectangle(paintWindow, (40,1 ), (140,65), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), colors[0], -1)
paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), colors[1], -1)
paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), colors[2], -1)
paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), colors[3], -1)
cv2.putText(paintWindow, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)

def printThreshold(thr):
    #this function is only used in the trackbar threshold thingy
    print("! Changed threshold to "+str(thr))

def centroid(largest_contour):
    moment = cv2.moments(largest_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None

def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        #if you have list of defects and a centre of hand
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None
    else:
        return None

cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', 0, 100, printThreshold)

while True:

    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    threshold = cv2.getTrackbarPos('trh1', 'trackbar') #the value from the trackbar

    lower_skin = np.array([0,37,0])
    upper_skin = np.array([17,154,255])

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    kernel = np.ones((15,15),np.float32)/225
    smoothed = cv2.filter2D(res,-1,kernel)
    median = cv2.medianBlur(res,15)
    #cv2.imshow('Median Blur',median)

    gray = cv2.cvtColor(cv2.cvtColor(median, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY) #converting the median blurred image to grayscale
    #cv2.imshow("gray", gray)

    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours is the list of numpy arrays of the countours found
    #cv2.imshow('threshold', thresh)


    number_of_contours = len(contours)
    largest_contour_area =  -1
    canvas = np.zeros(median.shape, np.uint8) #an empty canvas to draw on
    canvas = cv2.rectangle(canvas, (40,1), (140,65), (122,122,122), -1) #topleft, bottomRight, borderColour, -1 ==> fill rectangle
    canvas = cv2.rectangle(canvas, (160,1), (255,65), colors[0], -1)
    canvas = cv2.rectangle(canvas, (275,1), (370,65), colors[1], -1)
    canvas = cv2.rectangle(canvas, (390,1), (485,65), colors[2], -1)
    canvas = cv2.rectangle(canvas, (505,1), (600,65), colors[3], -1)
    cv2.putText(canvas, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)
    if number_of_contours > 0: #if you have found atleast 1 countours
        #loop through the contours and find the largest one which we consider is the hand
        for i in range(number_of_contours):
            current_contour_area = cv2.contourArea(contours[i])
            if current_contour_area > largest_contour_area:
                largest_contour_area = current_contour_area
                index_of_largest_contour = i

        largest_contour = contours[index_of_largest_contour]
        #largest_contour is the hand
        centre_of_hand = centroid(largest_contour)
        #centre_of_hand is the (x,y) of the centre of the hand

        cv2.drawContours(canvas, [largest_contour], 0, (0, 255, 0), 3) #thickness=cv2.FILLED
        #draw the hand
        #cv2.drawContours(canvas, [cv2.convexHull(largest_contour)], 0, (0, 0, 255), 3)
        #draw convex hull around hand
        cv2.circle(canvas, centre_of_hand, 5, [255, 0, 255], -1)
        #draws a purple dot at the centre of the hand

        hull = cv2.convexHull(largest_contour,returnPoints = False)
        defects = cv2.convexityDefects(largest_contour,hull)
        #two lines above are to find convexity defects

        if type(defects) != type(None):#this if statement finds the base of the fingers
            #if there are some defects

            number_of_fingers = 0
            #https://goo.gl/dkYhLs --> code below is from here
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(largest_contour[s][0])
                end = tuple(largest_contour[e][0])
                far = tuple(largest_contour[f][0])
                # cosine theorem
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    number_of_fingers += 1
                    cv2.circle(canvas,far,10,[255,0,0],-1) #find the fingers
            #the for loop puts the circles at the base of fingers

            fingerTip = farthest_point(defects, largest_contour, centre_of_hand)
            cv2.circle(canvas, fingerTip, 10, [255, 255, 0], -1) #fingertip is turquoise

            print("finger tip is in this format: X: %i Y: %i "%(fingerTip[0], fingerTip[1]))
            fingerTip_x = fingerTip[0]
            fingerTip_y = fingerTip[1]
            if user_currently_painting:
                if fingerTip_y <= 65: #if the user's fingertip is in the menu region
                    if 40 <= fingerTip_x <= 140: # user selecting CLEAR ALL
                        bpoints = [deque(maxlen=512)]
                        gpoints = [deque(maxlen=512)]
                        rpoints = [deque(maxlen=512)]
                        ypoints = [deque(maxlen=512)]

                        bindex = 0
                        gindex = 0
                        rindex = 0
                        yindex = 0

                        paintWindow[67:,:,:] = 255 #make the whole canvas white
                    elif 160 <= fingerTip_x <= 255:
                            colorIndex = 0 # Blue
                    elif 275 <= fingerTip_x <= 370:
                            colorIndex = 1 # Green
                    elif 390 <= fingerTip_x <= 485:
                            colorIndex = 2 # Red
                    elif 505 <= fingerTip_x <= 600:
                            colorIndex = 3 # Yellow
                else :
                    if colorIndex == 0:
                        bpoints[bindex].appendleft(fingerTip)
                    elif colorIndex == 1:
                        gpoints[gindex].appendleft(fingerTip)
                    elif colorIndex == 2:
                        rpoints[rindex].appendleft(fingerTip)
                    elif colorIndex == 3:
                        ypoints[yindex].appendleft(fingerTip)
            else:
                #pen_UP
                bpoints.append(deque(maxlen=512))
                bindex += 1
                gpoints.append(deque(maxlen=512))
                gindex += 1
                rpoints.append(deque(maxlen=512))
                rindex += 1
                ypoints.append(deque(maxlen=512))
                yindex += 1


            points = [bpoints, gpoints, rpoints, ypoints]
            for i in range(len(points)):
                for j in range(len(points[i])): #looping through the list of queues
                    length_of_dequeList = len(points[i][j])
                    for k in range(1, length_of_dequeList):#looping through the dequeList
                        if points[i][j][k - 1] is None or points[i][j][k] is None:
                            continue
                        #cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                        cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)
    else:
        print("on the screen show some text that says PLACE A SINGLE HAND ON SCREEN")


    cv2.imshow('output', canvas)
    cv2.imshow("Paint", paintWindow)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    if k == ord('p'):
        print("user wants to toggle the user_currently_painting value")
        user_currently_painting = not(user_currently_painting)


cv2.destroyAllWindows()
cap.release()
