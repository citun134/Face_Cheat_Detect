import numpy as np
from imutils.video import VideoStream
import cv2
import time
import os

import cvzone
from cvzone.FaceDetectionModule import FaceDetector

# Label
label = "none"

cap = VideoStream(src=0).start()
time.sleep(2.0)

haar_cascade = cv2.CascadeClassifier('D:/PyCharm/pythonProject/Face_Rec/haarcascade_frontalface_default.xml')

# Biến đếm, để chỉ lưu dữ liệu sau khoảng 60 frame, tránh lúc đầu chưa kịp cầm tiền lên
total = 0
while(True):
    # Capture frame-by-frame

    frame = cap.read()
    frame = cv2.flip(frame, 1)
    orig = frame.copy()

    frame = cv2.resize(frame, dsize=None, fx=0.5,fy=0.5)
    rects = haar_cascade.detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30))
    # loop over the face detections and draw them on the frame
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Hiển thị
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1) & 0xFF

    # Lưu dữ liệu
    if key == ord("k") and total <= 20:
        print("Số ảnh capture = ",total)
        # Tạo thư mục nếu chưa có
        if not os.path.exists('D:/PyCharm/pythonProject/Face_Rec/FaceDataset/train/' + str(label)):
            os.mkdir('D:/PyCharm/pythonProject/Face_Rec/FaceDataset/train/' + str(label))

        cv2.imwrite('D:/PyCharm/pythonProject/Face_Rec/FaceDataset/train/' + str(label) + "/" + str(total) + ".jpg",orig)
        total += 1

    if key == ord("k") and total > 20 and total <= 25:
        print("Số ảnh capture = ",total)
        # Tạo thư mục nếu chưa có
        if not os.path.exists('D:/PyCharm/pythonProject/Face_Rec/FaceDataset/val/' + str(label)):
            os.mkdir('D:/PyCharm/pythonProject/Face_Rec/FaceDataset/val/' + str(label))

        cv2.imwrite('D:/PyCharm/pythonProject/Face_Rec/FaceDataset/val/' + str(label) + "/" + str(total) + ".jpg",orig)
        total += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
# cap.release()
cv2.destroyAllWindows()