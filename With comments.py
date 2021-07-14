#Importing all modules
import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

# Importing and Reading the header images
folderPath = 'Header'
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    img = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(img)
print(len(overlayList))

#  Adding the 1st image to Header
header = overlayList[0]

#  Reading Webcam and setting size
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Using my HTM creating a object
detector = htm.handDetector(maxHands=1, detectionCon=0.825)

while True:
    # Reading hte Image of the Webcam and mirroring it
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #  Drawing the hands on the image using htm
    img = detector.findHands(img)

    # Getting the Landmarks
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        print(lmList)

    img[0:125, 0:1280] = header

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
