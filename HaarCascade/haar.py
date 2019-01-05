import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('face.xml')

eye_cascade = cv2.CascadeClassifier('eye.xml')


cap = cv2.VideoCapture(0)

cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 300,300) #resizing the window

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #draw a blue rect around the face
        roi_gray = gray[y:y+h, x:x+w] #this is the face region
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cv2.destroyAllWindows()
cap.release()
