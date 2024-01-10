import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import cv2

data = np.load("face_data.npy")

print(data.shape, data.dtype)

X = data[:, 1:].astype(int)
y = data[:, 0]

model = KNeighborsClassifier()
model.fit(X, y)

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

            out = model.predict([grey.flatten()])

            cv2.rectangle(frame, (x, y), (x+w, y+h), (100,150,70), 6)
            cv2.putText(frame, str(out[0]), (x ,y - 10), cv2.FONT_HERSHEY_COMPLEX, 2, (100,150,70), 6)
            print(out)
        cv2.imshow("My Screen", frame)
        

    # time.sleep stops everything.. not ableßß to grab anything from system.
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

