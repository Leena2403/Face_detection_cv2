import cv2
import dlib

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("/Users/leenagoyal/Downloads/ML/shape_predictor_68_face_landmarks.dat")


cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray, face)
        # print(landmarks.parts())
        nose = landmarks.parts()[28]
        # print(nose.x, nose.y)
        for point in landmarks.parts():
            cv2.circle(frame, (point.x, point.y), 2, (255, 0, 70), 4)

    # print(faces)
    if ret:
        cv2.imshow("My Screen", frame)

    # time.sleep stops everything.. not ableßß to grab anything from system.
    key = cv2.waitKey(1)

    if key == ord("q"):
        break
cap.release()    
cv2.destroyAllWindows()


