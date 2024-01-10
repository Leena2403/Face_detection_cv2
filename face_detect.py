
import cv2

cap = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("/Users/leenagoyal/Downloads/ML/haarcascade_frontalface_default.xml")

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
cap.release()    
cv2.destroyAllWindows()


