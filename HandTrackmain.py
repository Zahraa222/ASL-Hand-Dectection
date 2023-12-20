import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands # a formality to before before using this module
hands = mpHands.Hands(False)
mpDraw = mp.solutions.drawing_utils #to display red dots on each important hand landmark
prevTime = 0 
currentTime = 0


while True:
    success, img = cap.read()
    #send in rgb image to hands objecct
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    result = hands.process(imgRGB)
    #print(result.multi_hand_landmarks) #to detect position of the hand

    if result.multi_hand_landmarks: #if a hand is detected
        for handLms in result.multi_hand_landmarks:
            #extract the information of each visible hand (coordinates)
            for id,landmark in enumerate(handLms.landmark):
                print(id, landmark)
            



            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) #display original image with hand landmarks connected with HANDS_CONNECTIONS
    currentTime = time.time()
    fps = 1/(currentTime - prevTime)
    prevTime = currentTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)

    cv2.imshow("image", img)
    cv2.waitKey(1)
