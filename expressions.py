import cv2
import dlib
import numpy as np
import os 

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("/Users/leenagoyal/Downloads/ML/shape_predictor_68_face_landmarks.dat")


cap = cv2.VideoCapture(0)

mood = input("Enter your mood: ")

frames = []
outputs = []

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray, face)
        # print(landmarks.parts())
        nose = landmarks.parts()[28]
        # print(nose.x, nose.y)

        upper_lip = landmarks.part(62)
        lower_lip = landmarks.part(66)
        mouth_open_threshold = lower_lip.y - upper_lip.y

        lip_up = landmarks.parts()[62].y
        lip_down = landmarks.parts()[66].y

        mouth_open = (lower_lip.y - upper_lip.y) > 5

        # Display text based on mouth state
        text = "Mouth Open" if mouth_open else "Mouth Close"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # prints output in the console/terminal
        # if lip_down - lip_up > 5:
        #     print("Mouth is Open")
        # else:
        #     print("Mouth is Close")   
         
        for point in landmarks.parts():
            cv2.circle(frame, (point.x, point.y), 2, (255, 0, 70), 4)

        # flatten the face using numpy
        expression = np.array([[point.x - face.left(), point.y - face.top()] for point in landmarks.parts()[17:]])
        # print(expression.flatten())
    # print(faces)
        
    if ret:
        cv2.imshow("My Screen", frame)

    # time.sleep stops everything.. not ableßß to grab anything from system.
    key = cv2.waitKey(1)

    if key == ord("q"):
        break
    elif key == ord("c"):
        frames.append(expression.flatten())
        outputs.append([mood])

X = np.array(frames)
y = np.array(outputs)

data = np.hstack([y, X])
f_name = "face_mood.npy"

if os.path.exists(f_name):
    old = np.load(f_name) 
    data = np.vstack([old, data])
    
np.save(f_name, data)

# print(data.shape)
cap.release()    
cv2.destroyAllWindows()



