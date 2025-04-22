import numpy as np
import cv2
import time
import os

# Label
label = "test"

cap = cv2.VideoCapture(0)

# Biến đếm, để chỉ lưu dữ liệu sau khoảng 60 frame, tránh lúc đầu chưa kịp cầm tiền lên
i=0
while(True):
    # Capture frame-by-frame
    #
    i+=1
    ret, frame = cap.read()
    # time.sleep(0.5)
    if not ret:
        continue
    frame = cv2.resize(frame, dsize=None, fx=0.5,fy=0.5)

    # Hiển thị
    cv2.imshow('frame',frame)

    # Lưu dữ liệu
    if i>=0 and i<=50:
        print("Số ảnh capture = ",i)
        # Tạo thư mục nếu chưa có
        if not os.path.exists('D:/PyCharm/pythonProject/Face_Rec/FaceDataset/train/' + str(label)):
            os.mkdir('D:/PyCharm/pythonProject/Face_Rec/FaceDataset/train/' + str(label))

        cv2.imwrite('D:/PyCharm/pythonProject/Face_Rec/FaceDataset/train/' + str(label) + "/" + str(i) + ".jpg",frame)

    if i>=51 and i<=60:
        print("Số ảnh capture = ",i)
        # Tạo thư mục nếu chưa có
        if not os.path.exists('D:/PyCharm/pythonProject/Face_Rec/FaceDataset/val/' + str(label)):
            os.mkdir('D:/PyCharm/pythonProject/Face_Rec/FaceDataset/val/' + str(label))

        cv2.imwrite('D:/PyCharm/pythonProject/Face_Rec/FaceDataset/val/' + str(label) + "/" + str(i) + ".jpg",frame)

    # time.sleep(0.2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()