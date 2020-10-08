import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

current_emotion = "Neutral"
emotion_history = []
EMOTION_HISTORY_LENGTH = 4
EMOTION_FREQUENCY_THRESHHOLD = 2

#Called whenever the average emotion changes to anything other than neutral
def onEmotionChanged(emotion):
    print("Emotion changed to: " + emotion)

#Adds to the history of the last EMOTION_HISTORY_LENGTH emotions
#calls OnEmotionChanged if a specific emotion (which isn't Neutral) appears in the history more than EMOTION_FREQUENCY_THRESHHOLD times
def updateEmotion(emotion):
    global emotion_history
    global current_emotion
    global EMOTION_HISTORY_LENGTH

    if len(emotion_history) >= EMOTION_HISTORY_LENGTH:
        emotion_history.pop(0)

    emotion_history.append(emotion)
    mostFrequentEmotion = max(set(emotion_history), key = emotion_history.count)
    frequency = emotion_history.count(mostFrequentEmotion)

    if frequency >= EMOTION_FREQUENCY_THRESHHOLD and mostFrequentEmotion != current_emotion:
        current_emotion = mostFrequentEmotion
        if current_emotion != "Neutral":
            onEmotionChanged(mostFrequentEmotion)

def calcFaceArea(face):
    (x, y, width, height) = face
    return width * height

#Considered certain when 5 out of 6 of the outputs are all exactly zero.
#Note: If it is uncertain, at least one of the outputs will be about 5x10e-30, but the highest output will still be exactly one
def isCertain(predictionList):
    count = 0
    for probability in predictionList:
        if probability == 0:
            count += 1

    return count == len(predictionList) - 1

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

#Load the model
model.load_weights('model.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

images = {}
for emotion in ["Angry", "Happy", "Neutral", "Sad", "Surprised"]:
    images[emotion] = cv2.imread("img/" + emotion + ".png", cv2.IMREAD_COLOR)


# start the webcam feed
cap = cv2.VideoCapture(0)
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    if (len(faces) > 0):

        #find the largest face and ignore all others
        largestFace = faces[0]
        for face in faces:
            if calcFaceArea(face) > calcFaceArea(largestFace):
                largestFace = face

        #crop the image to fit the neural net inputs
        (x, y, w, h) = largestFace
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        #make a prediction
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))

        #manually override the glitchy predictions
        if (maxindex == 1 or maxindex == 2):
            maxindex = 4

        #if the net's decision is 100% certain, draw it on the screen
        if isCertain(prediction[0]):
            #print(str(emotion_dict[maxindex]) + str(prediction));
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2, cv2.LINE_AA)
            updateEmotion(emotion_dict[maxindex])
        else:
            updateEmotion("Neutral")

    #display the currently decided emotion in the top left
    if current_emotion != "Neutral":
        cv2.rectangle(frame, (0,0), (180, 45), (0, 0, 0), -1)
        cv2.putText(frame, current_emotion, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    #draw image to screen
    cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    cv2.imshow('image', images[current_emotion])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
