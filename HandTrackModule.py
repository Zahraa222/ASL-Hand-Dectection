import cv2
import mediapipe as mp
import time


class handDetection():
    def __init__(self, mode = False, maxHands = 2, modelComplexity = 1, detectionConfidence = 0.5, trackCon = 0.5):
        #create an object that has its own variable
        self.mode = mode #variable of the object (self.___)
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackCon = trackCon
        self.modelComplexity = modelComplexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity,self.detectionConfidence, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    

    def findHands(self, img, draw = True):
        #send in rgb image to hands objecct
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        self.result = self.hands.process(imgRGB)
        #print(result.multi_hand_landmarks) #to detect position of the hand

        if self.result.multi_hand_landmarks: #if a hand is detected
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS),self.mpHands.HAND_CONNECTIONS
        return img
    

    def findPosition(self, img, handNo = 0, draw=True):
        landmarkList = []

        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, landmark in enumerate(myHand.landmark):
                print(id, landmark)
                height, width, imgchannels = img.shape
                cposX, cposY = int(landmark.x * width), int(landmark.y * height)
                #print(id, cposX, cposY) 
                landmarkList.append([id, cposX, cposY])
                if draw:
                    cv2.circle(img, (cposX, cposY), 15, (255, 0, 255), cv2.FILLED)
        return landmarkList







#if we are running this script, then whatever will be in the main part will be like dummy code that will be used to showcase what can this module do
def main():
    prevTime = 0 
    currentTime = 0
    cap = cv2.VideoCapture(0)
    test = handDetection()
    

    while True:
        success, img = cap.read()
        img = test.findHands(img)
        lmlist = test.findPosition(img)
        if len(lmlist) !=0:
            print(lmlist[4]) #4 is the landmark of tip of thumb



        currentTime = time.time()
        fps = 1/(currentTime - prevTime)
        prevTime = currentTime
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)
        cv2.imshow("image", img)
        cv2.waitKey(1)

    
if __name__ == "__main__":
    main()