import numpy as np
import cv2 as cv
import time

# Pre trained detection models in python/site-packages/cv2/data
from cv2.data import haarcascades

face_cascade = cv.CascadeClassifier(haarcascades + "haarcascade_frontalface_alt.xml")
eye_cascade = cv.CascadeClassifier(haarcascades + "haarcascade_eye.xml")

nb_frame = 0

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
t_begin = time.time()
while True:
    
    nb_frame += 1
    
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly stop the capture
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
      
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    eyes = eye_cascade.detectMultiScale(frame, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    for (x, y, w, h) in eyes:
      cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

t_tot = time.time() - t_begin

avg_framerate = nb_frame / t_tot

print(avg_framerate)

cap.release()
cv.destroyAllWindows()