import cv2
import HandTrackModule as htm
from Classifier import Classification
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = htm.handDetection(maxHands=1) #working with 1 hand
classifier = Classification("TrainingModel/keras_model.h5", "TrainingModel/labels.txt")
offset = 20
#data available for all the alphabet except J and Z (motion signs)
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]


while True:
    success, img = cap.read()
    output = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        imgWhite = np.ones((300,300,3), np.uint8)*255 #300 * 200 matrix of a white image
        #startin height(y), ending height(y+h). starting width(x), ending width(x+h)
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]




        aspectRatio = h/w
        if aspectRatio > 1:
            k = 300 / h #stretch height to image size
            widthCalculated = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (widthCalculated, 300))
            imgResizeShape = imgResize.shape
            widthGap = math.ceil((300 - widthCalculated )/ 2)
            imgWhite[:, widthGap:widthCalculated+widthGap] = imgResize #width + gap to center the image, will generate cropped image with white background
        else:
            k = 300 / w #stretch width to image size
            heightCalculated = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (300, heightCalculated))
            imgResizeShape = imgResize.shape
            heightGap = math.ceil((300 - heightCalculated )/ 2)
            imgWhite[heightGap:heightCalculated+heightGap, :] = imgResize #width + gap to center the image, will generate cropped image with white background 
  
        prediction, index = classifier.predict(imgWhite)
        print(prediction, index)
        cv2.putText(output,labels[index], (x,y- 25), cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,0), 2)
        cv2.rectangle(output, (x - offset,y - offset), (x+w + offset, y+h+ offset), (0,255,255),2)


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        cv2.waitKey(1)
    cv2.imshow("Image", output)
    cv2.waitKey(1)

  