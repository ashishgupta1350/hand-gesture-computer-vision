import numpy as np
import cv2
import pandas as pd
import time


def preprocess(action_frame, frame):
    blur = cv2.GaussianBlur(action_frame, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # lower_color = np.array([0, 23, 82])
    # upper_color = np.array([40, 100, 255])
    lower_color = np.array([0, 10, 60], dtype="uint8")
    upper_color = np.array([20, 150, 255], dtype="uint8")

    mask = cv2.inRange(hsv, lower_color, upper_color)
    blur = cv2.medianBlur(mask, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hsv_d = cv2.dilate(blur, kernel)
    # cv2.morphologyEx(hsv_d, )
    hsv_d = cv2.dilate(blur, kernel)

    # hsv_d = cv2.morphologyEx(hsv_d, cv2.MORPH_CLOSE, kernel)
    return hsv_d


#
# cap = cv2.VideoCapture(0)
#
# if not cap.isOpened():
#     print("camera did not open!")
#     exit(1)
#
# _, frame = cap.read()
# print(frame.shape)
# while True:
#     _, frame = cap.read()
#     k = cv2.waitKey(1) & 0xFF
#
#     # we need a bounding box and region of interest for the hand
#     w, h, topx, topy = (150, 150, 50, 100)
#     roi = frame[topx:topx + 2 * h, topy:topy + 2 * w].copy()
#
#     frame = cv2.rectangle(frame, pt1=(50, 100), pt2=(topx + h, topy + w), thickness=3, color=(0, 255, 0))
#     frame = cv2.circle(frame, color=3, radius=2, center=(50, 1))
#
#     roi = preprocess(roi, frame)
#
#     cv2.imshow('ROI', roi)
#     cv2.imshow('Image', frame)
#     if k == 27 or k == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# Background Subraction
import cv2
import numpy as np

# # reframing
# cv2.namedWindow("First Frame", cv2.WINDOW_NORMAL)
# cv2.resizeWindow('First Frame', 800,600)
# cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Frame', 800,600)

# cap = cv2.VideoCapture("media/cameraRoll.mp4")
cap = cv2.VideoCapture(0)
frameCount = 0
imgCount = 0
imgString = 'image'
labels = []
imgs = []
labelCounter = 0
numLabels = 4

save = False

while (True):
    reading, image = cap.read()
    if not reading:
        break

    # code for thresholding the hand
    _, frame = cap.read()
    k = cv2.waitKey(1) & 0xFF

    # we need a bounding box and region of interest for the hand
    w, h, topx, topy = (100, 100, 50, 150)
    roi = frame[topx:topx + 2 * h, topy:topy + 2 * w].copy()

    frame = cv2.rectangle(frame, pt1=(50, 100), pt2=(topx + h, topy + w), thickness=3, color=(0, 255, 0))
    frame = cv2.circle(frame, color=3, radius=2, center=(50, 1))

    roi = preprocess(roi, frame)

    cv2.imshow('ROI', roi)
    cv2.imshow('Image', frame)
    if k == 27 or k == ord('q'):
        break

    frameCount += 1

    if (frameCount):

        if (imgCount % 100 == 0):
            labelCounter += 1
            if labelCounter == numLabels:
                break
            print('200 frames complete')

            print("starting in 5")
            time.sleep(2)
            print("starting in 3")
            time.sleep(1)
            print("starting in 2")
            time.sleep(1)
            print("starting in 1")
            time.sleep(1)
            print('started')

    imgCount += 1

    labels.append(labelCounter)
    roi = cv2.resize(roi, (50, 40), interpolation=1)

    # we can directly save the image as a csv file
    # as 50 * 40 image
    # imgArray = np.ravel(roi).copy()
    # imgArray = np.reshape((roi.shape[0]*roi.shape[1],1))
    imgArray = roi.reshape((2000,))
    print(imgArray.shape)
    imgs.append(imgArray)
    save = False
    if save:
        cv2.imwrite(
            'media/classifier_symbols/images/' + imgString + str(labelCounter) + str('-') + str(imgCount) + '.jpg', roi)
        print('Processing image: ', imgCount)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()

imgs = np.array(imgs)
labels = np.array(labels)
df = pd.DataFrame(labels)
df2 = pd.DataFrame(imgs)

df.to_csv('media/classifier_symbols/labels.csv', index=False, header=False)
df2.to_csv('media/classifier_symbols/images.csv', index=False, header=False)

print('end')
