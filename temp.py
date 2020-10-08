import cv2
import time

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)
start_time = time.time()
time.sleep(1)

counter = 0

# start the webcam feed
cap = cv2.VideoCapture(0)
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    counter += 1

    if time.time() - start_time > 1:
        start_time = time.time()
        print(counter)
        counter = 0


    cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
