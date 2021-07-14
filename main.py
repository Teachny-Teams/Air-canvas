import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

brushThickness = 15
folderPath = 'Header'
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    img = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(img)
print(len(overlayList))
eraserThickness = 100
header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(maxHands=1, detectionCon=0.75, trackCon=0.85)
xp, yp = 0, 0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)

    # Getting the Landmarks
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        #  Index
        x1, y1 = lmList[8][1:]
        # Middle
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        # print(fingers)

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

            print('Selection mode')
            # Y location for slection
            if y1 < 125:
                # Pink Color
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)

                # Blue Color
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)

                # Green Color
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)

                # Eraser
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1-25), (x2, y2+25),
                          drawColor, cv2.FILLED)

        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print('Drawing mode')

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1),
                         drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1),
                         drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _,imgInverse = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInverse)
    img = cv2.bitwise_or(img,imgCanvas)

    img[0:125, 0:1280] = header
    # img =cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow("Image", img)
    cv2.imshow("Image Canvas ", imgCanvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()