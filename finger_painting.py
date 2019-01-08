import cv2
import numpy as np
import math
from collections import deque

#painting stuff inspired by : #https://github.com/akshaychandra21/Webcam_Paint_OpenCV

webcam = cv2.VideoCapture(0) #accessing the webcam
cv2.namedWindow('output',cv2.WINDOW_NORMAL) #creating the window
cv2.resizeWindow('output', 600,600) #resizing the window



# Setup deques to store separate colours in separate arrays
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

#these indexes allow me to know what queue the current live painting should go into
bindex = 0
gindex = 0
rindex = 0
yindex = 0


colours = {"BLUE":(255, 0, 0), "GREEN":(0, 255, 0),"RED": (0, 0, 255), "YELLOW":(0, 255, 255)}
current_colour = "BLUE" #default colour is blue
user_currently_painting = False



'''
*******************
*******************
Setting up the painting canvas
*******************
*******************
'''
paintWindow = np.zeros((471,636,3)) + 255 #the white canvas for the paint application
#placing the rectangles for each colour at the top left
#cv2.rectangle(where_to_place_rectangle, top_left_corner, bottom_right_corner, colour, borderWidth)
paintWindow = cv2.rectangle(paintWindow, (40,1 ), (140,65), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), colours["BLUE"], -1) #negative border width => fill
paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), colours["GREEN"], -1)
paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), colours["RED"], -1)
paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), colours["YELLOW"], -1)
cv2.putText(paintWindow, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)


def printThreshold(thr):
    #this function is only used in the trackbar threshold thingy
    print("! Changed threshold to "+str(thr))
#function that finds the centre of mass of given contour
def centroid(largest_contour):

    moment = cv2.moments(largest_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None
#finds the furthest convexity defect from the centre of mass of a contour
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
def what_menu_button_pressed(fingerTip_x):
    if 40 <= fingerTip_x <= 140: # user selecting CLEAR ALL
        return "CLEAR ALL"
    elif 160 <= fingerTip_x <= 255:
        return "BLUE"
    elif 275 <= fingerTip_x <= 370:
        return "GREEN"
    elif 390 <= fingerTip_x <= 485:
        return "RED"
    elif 505 <= fingerTip_x <= 600:
        return "YELLOW"
def find_largest_contour(contours):
    number_of_contours = len(contours)
    largest_contour_area =  -1
    if number_of_contours > 0: #if you have found atleast 1 countours
        #loop through the contours and find the largest one which we consider is the hand
        for i in range(number_of_contours):
            current_contour_area = cv2.contourArea(contours[i])
            if current_contour_area > largest_contour_area:
                largest_contour_area = current_contour_area
                index_of_largest_contour = i

        largest_contour = contours[index_of_largest_contour]
        return largest_contour
    else:
        return None
def draw_dot_between_fingers(defects,largest_contour,canvas):
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
            #number_of_fingers += 1
            cv2.circle(canvas,far,10,[255,0,0],-1) #find the fingers

#creating the trackbar window
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', 0, 200, printThreshold)

while True:

    _, frame = webcam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    threshold = cv2.getTrackbarPos('trh1', 'trackbar') #the value from the trackbar

    lower_skin = np.array([0,37,0])
    upper_skin = np.array([17,154,255])

    mask = cv2.inRange(hsv, lower_skin, upper_skin) #masks everything so that only the "Skin" is left
    binary_image = cv2.bitwise_and(frame,frame, mask= mask) #makes the skin white and everything else black
    #more info on the kernal and smoothed can be found here : https://github.com/atduskgreg/opencv-processing-book/blob/master/book/filters/blur.md
    kernel = np.ones((15,15),np.float32)/225
    smoothed = cv2.filter2D(binary_image,-1,kernel)
    median = cv2.medianBlur(binary_image,15)

    gray = cv2.cvtColor(cv2.cvtColor(median, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY) #converting the median blurred image to grayscale

    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours is the list of numpy arrays of the countours found

    '''
    *******************
    *******************
    Setting up the frame to draw the hand on
    *******************
    *******************
    '''
    canvas = np.zeros(median.shape, np.uint8) #an empty canvas to draw on same size as the median image
    canvas = cv2.rectangle(canvas, (40,1), (140,65), (122,122,122), -1)
    canvas = cv2.rectangle(canvas, (160,1), (255,65), colours["BLUE"], -1)
    canvas = cv2.rectangle(canvas, (275,1), (370,65), colours["GREEN"], -1)
    canvas = cv2.rectangle(canvas, (390,1), (485,65), colours["RED"], -1)
    canvas = cv2.rectangle(canvas, (505,1), (600,65), colours["YELLOW"], -1)
    cv2.putText(canvas, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)

        largest_contour = find_largest_contour(contours)
        #largest_contour is taken to be the hand
        centre_of_hand = centroid(largest_contour)
        #centre_of_hand is the (x,y) of the centre of the hand
        cv2.drawContours(canvas, [largest_contour], 0, (0, 255, 0), 3)
        #draw the hand
        cv2.drawContours(canvas, [cv2.convexHull(largest_contour)], 0, (0, 0, 255), 3)
        #draw convex hull around hand
        cv2.circle(canvas, centre_of_hand, 5, [255, 0, 255], -1)
        #draws a purple dot at the centre of the hand

        hull = cv2.convexHull(largest_contour,returnPoints = False)
        defects = cv2.convexityDefects(largest_contour,hull)
        #two lines above are to find convexity defects


        if type(defects) != type(None): #if there are defects
            #if there are some defects
            #https://goo.gl/dkYhLs --> code below is from here
            draw_dot_between_fingers(defects,largest_contour,canvas)
            fingerTip = farthest_point(defects, largest_contour, centre_of_hand)
            cv2.circle(canvas, fingerTip, 10, [255, 255, 0], -1) #fingertip is turquoise
            print("finger tip is in this format: X: %i Y: %i "%(fingerTip[0], fingerTip[1]))
            fingerTip_x = fingerTip[0]
            fingerTip_y = fingerTip[1]
            if user_currently_painting:
                #pen_DOWN
                if fingerTip_y <= 65: #if the user's fingertip is in the menu region
                    menu_button_pressed = what_menu_button_pressed(fingerTip_x)
                    if menu_button_pressed == "CLEAR ALL": # user selecting CLEAR ALL
                        bpoints = [deque(maxlen=512)]
                        gpoints = [deque(maxlen=512)]
                        rpoints = [deque(maxlen=512)]
                        ypoints = [deque(maxlen=512)]

                        bindex = 0
                        gindex = 0
                        rindex = 0
                        yindex = 0

                        paintWindow[67:,:,:] = 255 #make the whole canvas white
                    else:
                        current_colour = menu_button_pressed
                else :
                    if current_colour == "BLUE":
                        bpoints[bindex].appendleft(fingerTip)
                    elif current_colour == "GREEN":
                        gpoints[gindex].appendleft(fingerTip)
                    elif current_colour == "RED":
                        rpoints[rindex].appendleft(fingerTip)
                    elif current_colour == "YELLOW":
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

            #this triple nested for loop is to draw the current drawings
            coloured_points = {"BLUE":bpoints, "GREEN":gpoints,"RED": rpoints, "YELLOW":ypoints}
            for key in coloured_points:
                for j in range(len(coloured_points[key])):
                    length_of_dequeList = len(coloured_points[key][j])
                    for k in range(1,length_of_dequeList):
                        if coloured_points[key][j][k - 1] is None or coloured_points[key][j][k] is None:
                            continue
                        cv2.line(paintWindow, coloured_points[key][j][k - 1], coloured_points[key][j][k], colours[key], 2)
    else:
        print("on the screen show some text that says PLACE A SINGLE HAND ON SCREEN")


    cv2.imshow('output', canvas)
    cv2.imshow("Paint", paintWindow)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    if k == ord('p'):
        #pen_up -> pen_down
        #pen_down -> pen_up
        user_currently_painting = not(user_currently_painting)


cv2.destroyAllWindows()
webcam.release()
