import cv2

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if classifier.empty():
    print("Error: Haar Cascade Classifier not loaded!")
else:
    print("Haar Cascade Classifier loaded successfully!")

while True:
    ret, frame = cap.read()
    if ret:
        gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30,30))
        print("Number of faces detected:", len(faces))

        for face in faces:
            x, y, w, h = face
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (250, 150, 20), 4)
        cv2.imshow("My window", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()