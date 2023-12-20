import cv2
import HandTrackModule as htm

cap = cv2.VideoCapture(0)
detector = htm.handDetection(maxHands=1)

while True:
    success, img = cap.read()
    hands = detector.findHands(img)

    img = detector.findHands(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)