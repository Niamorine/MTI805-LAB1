import cv2 as cv
from os import listdir
from cv2.data import haarcascades

imgs_folder = "./test_images/" # Source directory
result_folder = "./test_results_bis/" # Result directory

imgs_name = listdir(imgs_folder)
imgs_path = [imgs_folder + img_name for img_name in imgs_name]

faces_color = (255, 0, 0)
eyes_color = (0, 0, 255)

print(f"Found {len(imgs_path)} images")

face_cascade = cv.CascadeClassifier(haarcascades + "haarcascade_frontalface_alt.xml")
eye_cascade = cv.CascadeClassifier(haarcascades + "haarcascade_eye.xml")

for (index, img_path) in enumerate(imgs_path):
    img = cv.imread(img_path)

    # minNeighbors affects the quality of the detected faces (higher value results in less detections but with higher quality)
    faces = face_cascade.detectMultiScale(img, 1.05, 4)
    eyes = eye_cascade.detectMultiScale(img, 1.05, 4)

    # Draw rect for faces
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), faces_color, 25)
    
    # Draw rect for eyes
    for (x, y, w, h) in eyes:
        cv.rectangle(img, (x, y), (x+w, y+h), eyes_color, 25)
    
    # Save resulting image
    cv.imwrite(result_folder + imgs_name[imgs_path.index(img_path)], img)
    print(f"{index + 1} / {len(imgs_path)} done")
