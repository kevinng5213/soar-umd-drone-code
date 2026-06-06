import cv2

cascade_path = "/usr/share/opencv4/haarcascades/"


face_cascade = cv2.CascadeClassifier(cascade_path + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cascade_path + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cascade_path + "haarcascade_smile.xml")


cap = cv2.VideoCapture(0)
goofyahh = cv2.imread("/home/raspberry/Downloads/goofyahh.jpg")
cv2.namedWindow("GoofWin")
winopen = False
goofbool = False
while True:
    ret, frame = cap.read()
   
    if not ret:
        print("Failed to grab frame.")
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]


        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        goofbool = False
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 255, 0), 2)
            goofbool = True
   
    if goofbool:
        cv2.imshow("GoofWin", goofyahh)
        winopen = True
    else:
        if winopen:
            cv2.destroyWindow("GoofWin")
            winopen = False
    cv2.imshow('Smile Detection', frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        cap.release()
        cv2.destroyAllWindows()

