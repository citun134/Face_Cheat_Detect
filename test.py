import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pickle
import glob
import scipy.io

import cv2
import numpy as np
from l2cs import Pipeline, render
import torch
from pathlib import Path

cols = []
for pos in ['nose_', 'forehead_', 'left_eye_', 'mouth_left_', 'chin_', 'right_eye_', 'mouth_right_']:
    for dim in ('x', 'y'):
        cols.append(pos+dim)


def extract_features(img, face_mesh):
    NOSE = 1
    FOREHEAD = 10
    LEFT_EYE = 33
    MOUTH_LEFT = 61
    CHIN = 199
    RIGHT_EYE = 263
    MOUTH_RIGHT = 291

    result = face_mesh.process(img)
    face_features = []

    if result.multi_face_landmarks != None:
        for face_landmarks in result.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [FOREHEAD, NOSE, MOUTH_LEFT, MOUTH_RIGHT, CHIN, LEFT_EYE, RIGHT_EYE]:
                    face_features.append(lm.x)
                    face_features.append(lm.y)

    return face_features


def normalize(poses_df):
    normalized_df = poses_df.copy()

    for dim in ['x', 'y']:
        # Centerning around the nose
        for feature in ['forehead_' + dim, 'nose_' + dim, 'mouth_left_' + dim, 'mouth_right_' + dim, 'left_eye_' + dim,
                        'chin_' + dim, 'right_eye_' + dim]:
            normalized_df[feature] = poses_df[feature] - poses_df['nose_' + dim]

        # Scaling
        diff = normalized_df['mouth_right_' + dim] - normalized_df['left_eye_' + dim]
        for feature in ['forehead_' + dim, 'nose_' + dim, 'mouth_left_' + dim, 'mouth_right_' + dim, 'left_eye_' + dim,
                        'chin_' + dim, 'right_eye_' + dim]:
            normalized_df[feature] = normalized_df[feature] / diff

    return normalized_df


def draw_axes(img, pitch, yaw, roll, tx, ty, size=50):
    yaw = -yaw
    rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll]))[0].astype(np.float64)
    axes_points = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    axes_points = rotation_matrix @ axes_points
    axes_points = (axes_points[:2, :] * size).astype(int)
    axes_points[0, :] = axes_points[0, :] + tx
    axes_points[1, :] = axes_points[1, :] + ty

    new_img = img.copy()
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 0].ravel()), (255, 0, 0), 3)
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 1].ravel()), (0, 255, 0), 3)
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 2].ravel()), (0, 0, 255), 3)
    return new_img

if __name__ == "__main__":
    model = pickle.load(open('D:\PyCharm\pythonProject\Face_Rec\model.pkl', 'rb'))
    face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Khởi tạo pipeline cho Gaze Estimation
    gaze_pipeline = Pipeline(
        weights=Path(r"D:\PyCharm\pythonProject\Face_Rec\L2CSNet_gaze360.pkl"),
        arch="ResNet50",
        device=torch.device("cpu")
    )
    # Định nghĩa đường dẫn
    # image_path = "/kaggle/working/dang-may-tu-nhien.jpg"
    # image_path = "/kaggle/working/than-thai-la-gi.jpg"
    # image_path = "/kaggle/working/nguoi-phu-nu-mang-dac-diem-nao-tren-guong-mat-co-tuong-vuong-phudocx-1711441147638.jpeg"
    # image_path = "/kaggle/working/623d9bdb4b9ba2c5fb8a-1594749733042526546314.jpg"
    # image_path = "/kaggle/input/face-pro/anh1.png"
    # image_path = "/kaggle/working/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f68797374732f7079746f7263685f6d70696967617a655f64656d6f2f6d61737465722f6173736574732f726573756c74732f6d70696967617a655f696d61676530302e6a7067"
    # image_path = "/kaggle/working/1627398123764_50e0e34dc4.jpeg"
    # image_path = r"D:\PyCharm\pythonProject\Face_Rec\FaceDataset\val\to_van_tu\53.jpg"
    #
    # output_path = r"D:\PyCharm\pythonProject\Face_Rec\output.jpg"
    cap = cv2.VideoCapture(0)  # From Camera

    while (cap.isOpened()):
        ret, image_path = cap.read()
        if ret:
            # Đọc ảnh từ đường dẫn
            frame = cv2.flip(image_path, 1)
            # frame = cv2.imread(image_path)

            # Kiểm tra ảnh có tồn tại không
            if frame is None:
                raise FileNotFoundError(f"Không tìm thấy ảnh tại đường dẫn: {image_path}")

            # frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, img_c = frame.shape
            text = ''

            # Giả định các hàm extract_features, normalize, draw_axes đã được định nghĩa
            face_features = extract_features(frame, face_mesh)
            if len(face_features):
                face_features_df = pd.DataFrame([face_features], columns=cols)
                face_features_normalized = normalize(face_features_df)
                pitch_pred, yaw_pred, roll_pred = model.predict(face_features_normalized).ravel()
                nose_x = face_features_df['nose_x'].values * img_w
                nose_y = face_features_df['nose_y'].values * img_h
                frame = draw_axes(frame, pitch_pred, yaw_pred, roll_pred, nose_x, nose_y)

                # Xác định hướng dựa trên pitch và yaw
                if pitch_pred > 0.3:
                    text = 'Top'
                    if yaw_pred > 0.3:
                        text = 'Top Left'
                    elif yaw_pred < -0.3:
                        text = 'Top Right'
                elif pitch_pred < -0.3:
                    text = 'Bottom'
                    if yaw_pred > 0.3:
                        text = 'Bottom Left'
                    elif yaw_pred < -0.3:
                        text = 'Bottom Right'
                elif yaw_pred > 0.3:
                    text = 'Left'
                elif yaw_pred < -0.3:
                    text = 'Right'
                else:
                    text = 'Center'

            head_pose_text = f"head_pose: {text}"
            # Vẽ text lên ảnh
            cv2.putText(frame, head_pose_text, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Chuyển lại sang BGR để hiển thị và lưu
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Xử lý ảnh với mô hình Gaze Estimation
            results = gaze_pipeline.step(frame)
            frame = render(frame, results)
            print(results)

            # Lấy bounding box và thông tin pitch/yaw
            bbox = results.bboxes[0]  # Chỉ lấy bounding box đầu tiên
            print(bbox)
            pitch = results.pitch[0]  # Góc nhìn theo trục dọc
            yaw = results.yaw[0]  # Góc nhìn theo trục ngang

            # In thông tin về pitch và yaw
            print("pitch: {}".format(pitch))
            print("yaw: {}".format(yaw))

            # Lấy thông tin từ bounding box
            x_min, y_min, x_max, y_max = bbox
            # Tính toán chiều rộng và chiều cao của bounding box
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            # Tính toán tọa độ trung tâm của bounding box
            center_x = int(x_min + bbox_width / 2)
            center_y = int(y_min + bbox_height / 2)

            #######################
            # Mở rộng bounding box gấp đôi
            new_bbox_width = bbox_width * 2
            new_bbox_height = bbox_height * 2

            # Cập nhật lại các giá trị của x_min, y_min, x_max, y_max để mở rộng bounding box
            new_x_min = int(center_x - new_bbox_width / 2)
            new_y_min = int(center_y - new_bbox_height / 2)
            new_x_max = int(center_x + new_bbox_width / 2)
            new_y_max = int(center_y + new_bbox_height / 2)

            bbox_width_new = new_x_max - new_x_min
            bbox_height_new = new_y_max - new_y_min
            #######################

            # Tính toán tọa độ điểm gaze dựa vào công thức mới
            dx = -bbox_width * np.sin(pitch) * np.cos(yaw)  # Sử dụng công thức mới cho dx
            dy = -bbox_width * np.sin(yaw)  # Sử dụng công thức mới cho dy

            # Tính toán tọa độ điểm gaze trong bounding box
            gaze_point = (int(center_x + dx), int(center_y + dy))

            # Vẽ điểm gaze trên ảnh
            cv2.circle(frame, gaze_point, 10, (0, 0, 255), -1)

            # image_height, image_width = frame.shape[:2]

            # # Xác định vùng gaze thuộc hướng nào (9 hướng)
            # quadrants = [
            #     ("top_left", (0, 0, int(image_width / 3), int(image_height / 3))),
            #     ("top", (int(image_width / 3), 0, int(image_width * 2 / 3), int(image_height / 3))),
            #     ("top_right", (int(image_width * 2 / 3), 0, image_width, int(image_height / 3))),
            #     ("left", (0, int(image_height / 3), int(image_width / 3), int(image_height * 2 / 3))),
            #     ("center", (int(image_width / 3), int(image_height / 3), int(image_width * 2 / 3), int(image_height * 2 / 3))),
            #     ("right", (int(image_width * 2 / 3), int(image_height / 3), image_width, int(image_height * 2 / 3))),
            #     ("bottom_left", (0, int(image_height * 2 / 3), int(image_width / 3), image_height)),
            #     ("bottom", (int(image_width / 3), int(image_height * 2 / 3), int(image_width * 2 / 3), image_height)),
            #     ("bottom_right", (int(image_width * 2 / 3), int(image_height * 2 / 3), image_width, image_height)),
            # ]

            ##############
            image_height, image_width = bbox_height_new, bbox_width_new

            quadrants = [
                ("top_left", (new_x_min, new_y_min, new_x_min + int(image_width / 3), new_y_min + int(image_height / 3))),
                ("top", (new_x_min + int(image_width / 3), new_y_min, new_x_min + int(image_width * 2 / 3),
                         new_y_min + int(image_height / 3))),
                ("top_right", (new_x_min + int(image_width * 2 / 3), new_y_min, new_x_max, new_y_min + int(image_height / 3))),
                ("left", (new_x_min, new_y_min + int(image_height / 3), new_x_min + int(image_width / 3),
                          new_y_min + int(image_height * 2 / 3))),
                ("center", (
                new_x_min + int(image_width / 3), new_y_min + int(image_height / 3), new_x_min + int(image_width * 2 / 3),
                new_y_min + int(image_height * 2 / 3))),
                ("right", (new_x_min + int(image_width * 2 / 3), new_y_min + int(image_height / 3), new_x_max,
                           new_y_min + int(image_height * 2 / 3))),
                (
                "bottom_left", (new_x_min, new_y_min + int(image_height * 2 / 3), new_x_min + int(image_width / 3), new_y_max)),
                ("bottom", (
                new_x_min + int(image_width / 3), new_y_min + int(image_height * 2 / 3), new_x_min + int(image_width * 2 / 3),
                new_y_max)),
                ("bottom_right",
                 (new_x_min + int(image_width * 2 / 3), new_y_min + int(image_height * 2 / 3), new_x_max, new_y_max)),
            ]
            #############

            for quadrant, (x_min, y_min, x_max, y_max) in quadrants:
                print(f"Checking quadrant: {quadrant}, Bounds: {(x_min, y_min, x_max, y_max)}")
                if x_min <= gaze_point[0] <= x_max and y_min <= gaze_point[1] <= y_max:
                    print(f"✅ Gaze detected in quadrant: {quadrant}")
                    quadrant_gaze = f"Gaze: {quadrant}"
                    cv2.putText(frame, quadrant_gaze, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    break

            # Lưu ảnh đã được xử lý
            # cv2.imwrite(output_path, frame)

            # print(f"Ảnh đã được lưu tại: {output_path}")
            cv2.imshow('img', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
        else:
            break

    cv2.destroyAllWindows()
    cap.release()


    ######################
    # model = pickle.load(open('D:\PyCharm\pythonProject\Face_Rec\model.pkl', 'rb'))
    # # Khởi tạo FaceMesh
    # face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    #
    # # Đường dẫn đến ảnh
    # image_path = r"D:\PyCharm\pythonProject\Face_Rec\FaceDataset\val\to_van_tu\53.jpg"
    #
    # # Đọc ảnh từ đường dẫn
    # img = cv2.imread(image_path)
    # if img is None:
    #     print(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
    # else:
    #     # Chuyển sang RGB để xử lý với MediaPipe
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img_h, img_w, img_c = img.shape
    #     text = ''
    #
    #     # Giả định các hàm extract_features, normalize, draw_axes đã được định nghĩa
    #     face_features = extract_features(img, face_mesh)
    #     if len(face_features):
    #         face_features_df = pd.DataFrame([face_features], columns=cols)
    #         face_features_normalized = normalize(face_features_df)
    #         pitch_pred, yaw_pred, roll_pred = model.predict(face_features_normalized).ravel()
    #         nose_x = face_features_df['nose_x'].values * img_w
    #         nose_y = face_features_df['nose_y'].values * img_h
    #         img = draw_axes(img, pitch_pred, yaw_pred, roll_pred, nose_x, nose_y)
    #
    #         # Xác định hướng dựa trên pitch và yaw
    #         if pitch_pred > 0.3:
    #             text = 'Top'
    #             if yaw_pred > 0.3:
    #                 text = 'Top Left'
    #             elif yaw_pred < -0.3:
    #                 text = 'Top Right'
    #         elif pitch_pred < -0.3:
    #             text = 'Bottom'
    #             if yaw_pred > 0.3:
    #                 text = 'Bottom Left'
    #             elif yaw_pred < -0.3:
    #                 text = 'Bottom Right'
    #         elif yaw_pred > 0.3:
    #             text = 'Left'
    #         elif yaw_pred < -0.3:
    #             text = 'Right'
    #         else:
    #             text = 'Forward'
    #
    #     # Vẽ text lên ảnh
    #     cv2.putText(img, text, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    #
    #     # Chuyển lại sang BGR để hiển thị và lưu
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #
    #     # # Hiển thị ảnh
    #     # cv2.imshow('img', img)
    #     # cv2.waitKey(0)  # Chờ cho đến khi nhấn phím bất kỳ để đóng cửa sổ
    #     # cv2.destroyAllWindows()
    #
    #     # Lưu ảnh (tùy chọn)
    #     output_path = r"D:\PyCharm\pythonProject\Face_Rec\result.png"
    #     cv2.imwrite(output_path, img)
    #     print(f"Ảnh kết quả đã được lưu tại: {output_path}")
#############################


    # model = pickle.load(open('D:\PyCharm\pythonProject\Face_Rec\model.pkl', 'rb'))
    #
    # face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    #
    # cap = cv2.VideoCapture(0)  # From Camera
    #
    # while (cap.isOpened()):
    #
    #     # Take each frame
    #     ret, img = cap.read()
    #     if ret:
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         img = cv2.flip(img, 1)
    #         img_h, img_w, img_c = img.shape
    #         text = ''
    #
    #         face_features = extract_features(img, face_mesh)
    #         if len(face_features):
    #             face_features_df = pd.DataFrame([face_features], columns=cols)
    #             face_features_normalized = normalize(face_features_df)
    #             pitch_pred, yaw_pred, roll_pred = model.predict(face_features_normalized).ravel()
    #             nose_x = face_features_df['nose_x'].values * img_w
    #             nose_y = face_features_df['nose_y'].values * img_h
    #             img = draw_axes(img, pitch_pred, yaw_pred, roll_pred, nose_x, nose_y)
    #
    #             if pitch_pred > 0.3:
    #                 text = 'Top'
    #                 if yaw_pred > 0.3:
    #                     text = 'Top Left'
    #                 elif yaw_pred < -0.3:
    #                     text = 'Top Right'
    #             elif pitch_pred < -0.3:
    #                 text = 'Bottom'
    #                 if yaw_pred > 0.3:
    #                     text = 'Bottom Left'
    #                 elif yaw_pred < -0.3:
    #                     text = 'Bottom Right'
    #             elif yaw_pred > 0.3:
    #                 text = 'Left'
    #             elif yaw_pred < -0.3:
    #                 text = 'Right'
    #             else:
    #                 text = 'Forward'
    #
    #         cv2.putText(img, text, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    #
    #         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #         cv2.imshow('img', img)
    #         k = cv2.waitKey(1) & 0xFF
    #         if k == ord("q"):
    #             break
    #     else:
    #         break
    #
    # cv2.destroyAllWindows()
    # cap.release()