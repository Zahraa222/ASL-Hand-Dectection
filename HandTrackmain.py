import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands # a formality to before before using this module
hands = mpHands.Hands(False)
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    #send in rgb image to hands objecct
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    result = hands.process(imgRGB)
    #print(result.multi_hand_landmarks) #to detect position of the hand

    if result.multi_hand_landmarks: #if a hand is detected
        for handLms in result.multi_hand_landmarks:
            #extract the information of each visible hand
            mpDraw.draw_landmarks(img, handLms) #display original image

    cv2.imshow("image", img)
    cv2.waitKey(1)
