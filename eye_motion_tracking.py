import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture("new.mp4")

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

while True:
    ret, img = cap.read()
    if ret is False:
        break
    img = imutils.rotate(img, 90)
    frame = img
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    eyes = eye_cascade.detectMultiScale(gray)
    # print(eyes)
    a = eyes[0]
    b = eyes[1]
    if(a[0] < b[0]):
        a = b
    (ex,ey,ew,eh) = a

    # print((ex,ey,ew,eh))
    cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    roi = frame[ey: ey+eh, ex: ex+ew]
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    _, threshold = cv2.threshold(gray_roi, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break

    cv2.imshow("Threshold", threshold)
    cv2.imshow("gray roi", gray_roi)
    cv2.imshow("Roi", roi)
    cv2.imshow("Main", frame)
    key = cv2.waitKey(100)
    if key == 27:
        break

cv2.destroyAllWindows()
