import cv2
import numpy as np
import tensorflow


class Classification:
    #classifies images of the pre-trained keras model
    def __init__(self, pathOfModel, labelsPath = None):

        self.pathOfModel = pathOfModel #directory to teh keras model
        #load model
        self.model = tensorflow.keras.models.load_model(self.pathOfModel)
        self.data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)

        self.labels_Path = labelsPath

        #read and store labels when file is given
        if self.labels_Path:
            label_file = open(self.labels_Path, "r")
            self.list_labels = [line.strip() for line in label_file]
            label_file.close()
        else:
            print("no labels identified")
    
    #classifies and displayes the result on the image
    def predict(self, img, draw=True, pos=(50,50), scale=2, color = (255,0,0)):
        givenImage = cv2.resize(img, (224,224))
        img_array = np.asarray(givenImage)
        normalizedArray = (img_array.astype(np.float32) / 127.0) - 1

        #load image to array
        self.data[0] = normalizedArray

        #make a prediction
        prediction = self.model.predict(self.data)
        indexVal = np.argmax(prediction)

        #draw on image prediction result
        if draw and self.labels_Path:
            cv2.putText(img, str(self.list_labels[indexVal]), pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, 2)
        return list(prediction[0]), indexVal

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    path = "."
    maskClassifier = (f'{path}/keras_model.h5', f'{path}/labels.txt')

    while True:
        _, img = cap.read()
        prediction = maskClassifier.predict(img)
        print(prediction) #result of classification
        cv2.imshow("Image", img)
        cv2.waitKey(1)

        