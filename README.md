This project is focused on developing a hand gesture recognition system using Python. It leverages computer vision techniques to detect hand gestures and classify them into different categories. The application is capable of recognizing various hand signs corresponding to the alphabet (excluding J and Z, which are motion signs).
The project is run in the **test.py** file.


**Dependencies**

To run this project, you need to install the following libraries:

**opencv-python** for image processing.

**mediapipe** for hand tracking.

**numpy** for numerical operations.

**tensorflow** for running the classification model.

**math** for mathematical calculations.

The mediapipe library is used to extract the position of the and using numbered keypoints. 
The hand is annotated with 21 points, each corresponding to a specific part of the hand.
![image](https://github.com/Zahraa222/ASL-Hand-Dectection/assets/66334839/902518a2-13c4-4db4-9f5c-9d09478fb6ea)

**Model Training and Data Preparation**
The classification model is trained on images representing different hand signs. I used Google's Teachable Machine for training the model and storing the data, where each folder of hand gestures was assigned a corresponding alphabet letter in Teachable Machine. The model and its labels are stored in the TrainingModel directory.
