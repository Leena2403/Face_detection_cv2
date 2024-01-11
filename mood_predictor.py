import cv2
import dlib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/leenagoyal/Downloads/ML/shape_predictor_68_face_landmarks.dat")

# Load the trained model
data = np.load("face_mood.npy")

X = data[:, 1:].astype(int)
y = data[:, 0]

model = KNeighborsClassifier()
model.fit(X, y)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(grey)
    for face in faces:
        landmarks = predictor(grey, face)
        nose = landmarks.parts()[27]

        expression = np.array([[point.x - face.left(), point.y - face.top()] for point in landmarks.parts()[17:]])

        print(model.predict(expression.flatten().reshape(1, -1)))


    if ret:
        cv2.imshow("My screen", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()    
cv2.destroyAllWindows()
