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
    fps = counter / (time.time() - start_time)
    print(fps)
    counter += 1
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
