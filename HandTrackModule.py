import cv2
import mediapipe as mp
import math


class handDetection():
    def __init__(self, staticMode = False, maxHands = 2, modelComplexity = 1, detectionConfidence = 0.5, trackCon = 0.5):
        #create an object that has its own variable,variable of the object (self.___)
        self.staticMode = staticMode #detection is processed per image (static mode)
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackCon = trackCon
        self.modelComplexity = modelComplexity
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.staticMode, self.maxHands, self.modelComplexity,self.detectionConfidence, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20] #ID numbers of each fingertip
        self.fingers = []
        self.landmarkList = []
    

    def findHands(self, img, draw = True, flipType = True):
        #send in rgb image to hands objecct
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        self.result = self.hands.process(imgRGB)
        #print(result.multi_hand_landmarks) #to detect position of the hand

        allHands = []
        height, width, imgchannels = img.shape

        if self.result.multi_hand_landmarks: #if a hand is detected
            for handType, handLandmarks in zip(self.result.multi_handedness, self.result.multi_hand_landmarks):
                myHand = {}
                mylandmarkList = []
                xposList = []
                yposList = []
                zposList = []

                for id, landmark in enumerate(handLandmarks.landmark):
                    posX, posY, posZ = int(landmark.x * width), int(landmark.y * height), int(landmark.z * width)
                    mylandmarkList.append([posX,posY,posZ])
                    xposList.append(posX)
                    yposList.append(posY)
                    zposList.append(posZ)
                
                #border box (bbox)
                    xmin, xmax = min(xposList), max(xposList)
                    ymin, ymax = min(yposList), max(yposList)
                    boxWidth, boxHeight = xmax - xmin, ymax - ymin
                    bbox = xmin, ymin, boxWidth, boxHeight
                    cx, cy = bbox[0] + (bbox[2] // 2),  bbox[1] + (bbox[3] // 2) #center x and y coordinate

                    myHand["landmark List"] = mylandmarkList
                    myHand["bbox"] = bbox
                    myHand["center"] = (cx,cy)

                    if flipType: #determine which hand is being displayed, left or right
                        if handType.classification[0].label == "Right":
                            myHand["Type"] = "Left"
                        else:
                            myHand["Type"] = "Right"

                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20), (255, 255, 0), 2) #draw a border sround the hand
                    cv2.putText(img, myHand["Type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2) #write text on top of box to indicate which arm is being raised
        return allHands, img
    
    def raisedFingers(self, myHand): #returns a list of raised fingers
        fingers = []
        myHandType = myHand["Type"]
        mylandmarkList = myHand["landmark List"]

        if self.result.multi_hand_landmarks:

            #first detect thumb (id=0)
            if myHandType == "Right":
                if mylandmarkList[self.tipIds[0]][0] > mylandmarkList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else: #left hand
                if mylandmarkList[self.tipIds[0]][0] < mylandmarkList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        #the rest of the fingers
            for id in range(1,5):
                if mylandmarkList[self.tipIds[id]][1] < mylandmarkList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers



    def distance(self, point1, point2, img = None, color = (255,255,0), scale = 5): #distance between two landmarks
        x1,y1 = point1
        x2,y2 = point2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        pointInfo = (x1, y1, x2, y2, cx, cy)

        if img != None:
            cv2.circle(img, (x1,y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x2,y2), scale, color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2,y2), color, max(1,scale //3))
            cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)
        return length, pointInfo, img







#if we are running this script, then whatever will be in the main part will be like dummy code that will be used to showcase what can this module do
def main():


    cap = cv2.VideoCapture(0)

    #initialize class
    test = handDetection(staticMode=False, maxHands=2,modelComplexity=1, detectionConfidence=0.5, trackCon=0.5)
    
    #Get frames
    while True:
        #success = frame is captured
        success, img = cap.read()
        #draws landmarks
        #flips image for better detection
        hands, img = test.findHands(img, draw=True, flipType=True)

        if hands:
            hand1 = hands[0]
            landmarkList1 = hand1["landmark List"] #generate list of all landmarks on hand1
            bbox1 = hand1["bbox"] #bounding box
            center1 = hand1["center"] #center coordinates
            handType1 = hand1["type"] #left or right hand


            #now that we the information collected of the hand being raised
            #we'll determine how many fingers are raised
            fingers1 = test.fingersUp(hand1)

             # Calculate distance between specific landmarks on the first hand and draw it on the image
            length, info, img = test.findDistance(landmarkList1[8][0:2], landmarkList1[12][0:2], img, color=(255, 0, 255), scale=10)

            #------------------------------------------------------------------------------------------------------------------------------------
            #repeat if a second hand is detected
            if len(hands == 2):
             hand2 = hands[1]
            landmarkList1 = hand2["landmark List"] #generate list of all landmarks on hand2
            bbox1 = hand2["bbox"] #bounding box
            center1 = hand2["center"] #center coordinates
            handType1 = hand2["type"] #left or right hand


            #now that we the information collected of the hand being raised
            #we'll determine how many fingers are raised
            fingers1 = test.fingersUp(hand2)

             # Calculate distance between specific landmarks on the first hand and draw it on the image
            length, info, img = test.findDistance(landmarkList1[8][0:2], landmarkList1[12][0:2], img, color=(255, 0, 255), scale=10)

    #display Image
    cv2.imshow("image", img)
    cv2.waitKey(1)

    
if __name__ == "__main__":
    main()