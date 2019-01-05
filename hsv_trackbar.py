#colour tracking

import cv2
import numpy as np

def nothing(x):
    pass

webcam = cv2.VideoCapture(0) #this would be the camera feed from the hololens
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 300,300)
cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
cv2.resizeWindow('mask', 300,300)
cv2.namedWindow('result',cv2.WINDOW_NORMAL)
cv2.resizeWindow('result', 300,300)

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)


while True:
    _, frame = webcam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower_skin = np.array([l_h, l_s, l_v])
    upper_skin = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("frame", frame) #show the frame which is 300x300
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

    key = cv2.waitKey(1)
    if key == 27: #the esc key
        break

cap.release()
cv2.destroyAllWindows()
