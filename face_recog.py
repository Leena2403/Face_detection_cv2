

import cv2
import os
import numpy as np

# data = np.load("face_data.npy")
# print(data.shape)

cap = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("/Users/leenagoyal/Downloads/ML/haarcascade_frontalface_default.xml")

name = input("Enter your name: ")

frames = []
outputs = []


while True:
    ret, frame = cap.read()
    if ret:
        faces = detector.detectMultiScale(frame)
        for face in faces:
            x,y,w,h = face
            cut = frame[y:y+h, x:x+w]
            fi = cv2.resize(cut, (100, 100))
            grey = cv2.cvtColor(fi, cv2.COLOR_BGR2GRAY)
            cv2.imshow("My Face", grey)
        cv2.imshow("My Screen", frame)
        

    # time.sleep stops everything.. not ableßß to grab anything from system.
    key = cv2.waitKey(1)

    if key == ord("q"):
        break
    if key == ord("c"):
        frames.append(grey.flatten())
        outputs.append([name])
        # save_path = "/Users/leenagoyal/Downloads/ML"
        # filename = name + ".jpeg"
        # cv2.imwrite(name + ".jpg", frame)

X = np.array(frames)
y = np.array(outputs)

data = np.hstack([y, X])
f_name = "face_data.npy"

if os.path.exists(f_name):
    old = np.load(f_name)
    data = np.vstack([old, data])
    
np.save(f_name, data)

# print(data.shape)
cap.release()    
cv2.destroyAllWindows()