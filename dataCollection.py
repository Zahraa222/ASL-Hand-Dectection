import cv2
import HandTrackModule as htm

cap = cv2.VideoCapture(0)
detector = htm.handDetection(maxHands=1)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        #startin height(y), ending height(y+h). starting width(x), ending width(x+h)
        imgCrop = img[y:y+h, x:x+w]
        cv2.imshow("ImageCrop", imgCrop)
        cv2.waitKey(1)
    cv2.imshow("ImageCrop", img)
    cv2.waitKey(1)