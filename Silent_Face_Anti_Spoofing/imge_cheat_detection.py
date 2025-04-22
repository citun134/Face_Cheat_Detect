# #############################################
# face detection with mtcnn on live cam feed  #
###############################################
import warnings
warnings.filterwarnings("ignore")
import os, sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
# from keras.models import load_model
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer, LabelEncoder
import mediapipe as mp
import pandas as pd

import pickle

import cv2
from l2cs import Pipeline, render
import torch
from pathlib import Path

###
# sys.path.append(os.path.join(os.getcwd(), "Silent_Face_Anti_Spoofing"))
# from Silent_Face_Anti_Spoofing.test import test
from test import test

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
# from Silent_Face_Anti_Spoofing.src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

from screeninfo import get_monitors




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


class FaceDetectors:

    def __init__(self):
        self.facenet_model = FaceNet()
        self.svm_model = pickle.load(open("D:/PyCharm/pythonProject/Face_Rec/FaceDataset/SVM_classifier.sav", 'rb'))
        self.data = np.load('D:/PyCharm/pythonProject/Face_Rec/FaceDataset/faces_dataset_embeddings.npz')
        # object to the MTCNN detector class
        self.detector = MTCNN()

    def face_mtcnn_extractor(self, frame):
        """Methods takes in frames from video, extracts and returns faces from them"""
        # Use MTCNN to detect faces in each frame of the video
        result = self.detector.detect_faces(frame)
        return result

    def face_localizer(self, person):
        """Method takes the extracted faces and returns the coordinates"""
        # 1. Get the coordinates of the face
        bounding_box = person['box']
        x1, y1 = abs(bounding_box[0]), abs(bounding_box[1])
        width, height = bounding_box[2], bounding_box[3]
        x2, y2 = x1 + width, y1 + height
        return x1, y1, x2, y2, width, height

    def face_preprocessor(self, frame, x1, y1, x2, y2, required_size=(160, 160)):
        """Method takes in frame, face coordinates and returns preprocessed image"""
        # 1. extract the face pixels
        face = frame[y1:y2, x1:x2]
        # 2. resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)

        # # 3. scale pixel values
        # face_pixels = face_array.astype('float32')
        # # 4. standardize pixel values across channels (global)
        # mean, std = face_pixels.mean(), face_pixels.std()
        # face_pixels = (face_pixels - mean) / std

        # 5. transform face into one sample
        # samples = np.expand_dims(face_pixels, axis=0)
        samples = np.expand_dims(face_array, axis=0)

        # 6. get face embedding
        # yhat = self.facenet_model.predict(samples)

        yhat = self.facenet_model.embeddings(samples)

        face_embedded = yhat[0]
        # 7. normalize input vectors
        in_encoder = Normalizer(norm='l2')
        X = in_encoder.transform(face_embedded.reshape(1, -1))
        return X

    def face_svm_classifier(self, X):
        """Methods takes in preprocessed images ,classifies and returns predicted Class label and probability"""
        # predict
        yhat = self.svm_model.predict(X)
        label = yhat[0]
        yhat_prob = self.svm_model.predict_proba(X)
        probability = round(yhat_prob[0][label], 2)
        trainy = self.data['arr_1']
        # predicted label decoder
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        predicted_class_label = out_encoder.inverse_transform(yhat)
        label = predicted_class_label[0]
        return label, str(probability)

    def face_detector(self, image_path):
        """Method classifies faces on live cam feed
           Class labels : sai_ram, donald_trump,narendra_modi, virat_koli"""
        frame = cv2.imread(image_path) ## --> nếu sử dụng lưu ảnh tạm trong thư mục
        # frame = image_path ## --> nếu sử dụng đọc ảnh từ api

        # monitor = get_monitors()[0]
        # screen_width = monitor.width
        # screen_height = monitor.height

        # Tính 2/3 chiều rộng và chiều cao
        # frame_width = int(screen_width * (7 / 8))
        # frame_height = int(screen_height * (7 / 8))
        # print(f"frame_width{frame_width}, frame_height{frame_height}")

        frame_height, frame_width = frame.shape[:2]

        print("Chiều rộng:", frame_width)
        print("Chiều cao:", frame_height)

        if frame is None:
            return {"error": f"Không thể đọc ảnh từ {image_path}"}

###############################################
        # def equalize_brightness(image):
        #     # Tách các kênh màu
        #     r, g, b = cv2.split(image)
        #
        #     # Cân bằng lược đồ màu cho từng kênh màu
        #     r_equalized = cv2.equalizeHist(r)
        #     g_equalized = cv2.equalizeHist(g)
        #     b_equalized = cv2.equalizeHist(b)
        #
        #     # Ghép các kênh màu đã cân bằng lại thành ảnh RGB
        #     equalized_image = cv2.merge([r_equalized, g_equalized, b_equalized])
        #
        #     return equalized_image

        def equalize_brightness(image):
            # Tách các kênh màu
            b, g, r = cv2.split(image)

            # Tạo một đối tượng CLAHE (Contrast Limited Adaptive Histogram Equalization) cho từng kênh màu
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

            # Áp dụng CLAHE cho từng kênh màu
            b_equalized = clahe.apply(b)
            g_equalized = clahe.apply(g)
            r_equalized = clahe.apply(r)

            # Ghép các kênh màu đã cân bằng trở lại thành ảnh RGB
            equalized_image = cv2.merge([b_equalized, g_equalized, r_equalized])

            return equalized_image




        ###############################################

        model = pickle.load(open('D:\PyCharm\pythonProject\Face_Rec\model.pkl', 'rb'))
        face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Khởi tạo pipeline cho Gaze Estimation
        gaze_pipeline = Pipeline(
            weights=Path(r"D:\PyCharm\pythonProject\Face_Rec\L2CSNet_gaze360.pkl"),
            arch="ResNet50",
            device=torch.device("cpu")
        )

        target_fps = 30
        frame_time = 1 / target_fps

        haar_cascade = cv2.CascadeClassifier('D:/PyCharm/pythonProject/Face_Rec/haarcascade_frontalface_default.xml')

        # while True:
        start_time = time.time()  # Bắt đầu đo thời gian
        # Capture frame-by-frame
        # __, frame1 = cap.read()

        frame = cv2.flip(frame, 1)
        # frame = equalize_brightness(frame)

        result_dict = {}


########################################################
        #### Xử lý head pose and gaze
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
            result_dict["head_pose"] = text

        head_pose_text = f"head_pose: {text}"
        # Vẽ text lên ảnh
        cv2.putText(frame, head_pose_text, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Chuyển lại sang BGR để hiển thị và lưu
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Xử lý ảnh với mô hình Gaze Estimation
        # results = gaze_pipeline.step(frame)

        try:
            results = gaze_pipeline.step(frame)
        except ValueError as e:
            if "need at least one array to stack" in str(e):
                print("Warning: Không phát hiện được khuôn mặt, bỏ qua tính toán gaze.")
                results = None
            else:
                raise

        if results is not None:
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

            #### tính khoảng cách mặt###########
            KNOWN_DISTANCE = 50.0  # cm
            KNOWN_WIDTH = 15.0  # cm
            face_width_pixels = bbox_width
            # focal_length = (face_width_pixels * KNOWN_DISTANCE) / KNOWN_WIDTH

            if frame_width <= 400:
                focal_length = 350  # Dành cho ảnh rất nhỏ
                print("Kích thước nhỏ → focal ~ 400px")

            elif 400 < frame_width <= 1000:
                focal_length = 1000  # Phù hợp cho ảnh ~640x480
                print("Kích thước vừa → focal ~ 800px")

            elif 1000 < frame_width <= 1280:
                focal_length = 1400  # Phù hợp cho HD 720p
                print("Kích thước HD → focal ~ 1300px")

            else:
                focal_length = 2000  # Phù hợp cho FullHD trở lên
                print("Kích thước lớn → focal ~ 2000px")

            print(f"face_width_pixels {face_width_pixels}")
            print(f"Focal length (hiệu chuẩn): {focal_length:.2f} pixels")

            def calculate_face_distance(face_width_pixels, known_width=15.0, focal_length=focal_length):
                if face_width_pixels <= 0:
                    return None
                return (known_width * focal_length) / face_width_pixels

            distance = calculate_face_distance(face_width_pixels)
            # distance = (KNOWN_WIDTH * focal_length) / face_width_pixels

            result_dict = {"distance": float(f"{distance:.2f}")}
            # cv2.putText(frame, f"{distance:.2f} cm", (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if distance:
                cv2.putText(frame, f"{distance:.2f} cm", (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                print(f"Khoảng cách: {distance:.2f} cm")
            else:
                print("Không thể tính khoảng cách")

            #### tính khoảng cách mặt###########

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

            # Tính toán tọa độ điểm gaze dựa vào công thức mới
            dx = -bbox_width * np.sin(pitch) * np.cos(yaw)  # Sử dụng công thức mới cho dx
            dy = -bbox_width * np.sin(yaw)  # Sử dụng công thức mới cho dy

            ############### gaze mắt 4 góc màn hình
            # monitor = get_monitors()[0]
            # screen_width_laptop = monitor.width
            # screen_height_laptop = monitor.height
            # print(f"Width: {screen_width_laptop}, Height: {screen_height_laptop}")
            screen_width = frame_width  # pixels
            screen_height = frame_height

            if distance > 60:
                print("Khoảng cách hợp lý (60cm - vô hạn cm)")
                lrate = 1.7
                dx_new = -screen_width * np.sin(pitch) * np.cos(yaw) * lrate  # Khoảng cách ngang từ trung tâm
                dy_new = -screen_height * np.sin(yaw) * lrate
                gaze_point_new = (int(center_x + dx_new), int(center_y + dy_new))

            if 45 <= distance <= 60:
                print("Khoảng cách hợp lý (45cm - 60cm)")
                lrate = 1.48
                dx_new = -screen_width * np.sin(pitch) * np.cos(yaw) * lrate  # Khoảng cách ngang từ trung tâm
                dy_new = -screen_height * np.sin(yaw) * lrate
                gaze_point_new = (int(center_x + dx_new), int(center_y + dy_new))

            elif 40 <= distance < 45:
                print("Khoảng cách hợp lý (30cm - 44cm)")
                lrate = 1.15
                dx_new = -screen_width * np.sin(pitch) * np.cos(yaw) * lrate  # Khoảng cách ngang từ trung tâm
                dy_new = -screen_height * np.sin(yaw) * lrate
                gaze_point_new = (int(center_x + dx_new), int(center_y + dy_new))
            elif 35 <= distance < 40:
                print("Khoảng cách hợp lý (35cm - 40cm)")
                lrate = 1.05
                dx_new = -screen_width * np.sin(pitch) * np.cos(yaw) * lrate  # Khoảng cách ngang từ trung tâm
                dy_new = -screen_height * np.sin(yaw) * lrate
                gaze_point_new = (int(center_x + dx_new), int(center_y + dy_new))
            elif distance < 35:
                print("Khoảng cách hợp lý (0cm - 35cm)")
                lrate = 1
                dx_new = -screen_width * np.sin(pitch) * np.cos(yaw) * lrate  # Khoảng cách ngang từ trung tâm
                dy_new = -screen_height * np.sin(yaw) * lrate
                gaze_point_new = (int(center_x + dx_new), int(center_y + dy_new))

            # lrate = 1.45
            # dx_new = -screen_width * np.sin(pitch) * np.cos(yaw) * lrate  # Khoảng cách ngang từ trung tâm
            # dy_new = -screen_height * np.sin(yaw) * lrate
            # gaze_point_new = (int(center_x + dx_new), int(center_y + dy_new))
            cv2.circle(frame, gaze_point_new, 15, (0, 255, 0), -1)

            corner_size = 0.25
            corner_width = int(screen_width * corner_size)
            corner_height = int(screen_height * corner_size)
            corners = {
                "top_left": (0, 0, corner_width, corner_height),
                "top_right": (screen_width - corner_width, 0, screen_width, corner_height),
                "bottom_left": (0, screen_height - corner_height, corner_width, screen_height),
                "bottom_right": (
                    screen_width - corner_width, screen_height - corner_height, screen_width, screen_height)
            }

            corner_detected = "none"
            colors = {
                "top_left": (0, 255, 0),  # Xanh lá
                "top_right": (0, 0, 255),  # Đỏ
                "bottom_left": (255, 0, 0),  # Xanh dương
                "bottom_right": (0, 255, 255)  # Vàng
            }

            for corner_name, (x_min, y_min, x_max, y_max) in corners.items():
                # Vẽ hình chữ nhật cho mỗi góc
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), colors[corner_name], 2)
                # Thêm nhãn tên góc
                cv2.putText(frame, corner_name, (x_min + 5, y_min + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[corner_name], 2)

            ############### gaze mắt 4 góc màn hình


            # Tính toán tọa độ điểm gaze trong bounding box
            gaze_point = (int(center_x + dx), int(center_y + dy))

            # Vẽ điểm gaze trên ảnh
            cv2.circle(frame, gaze_point, 10, (0, 0, 255), -1)

            image_height, image_width = bbox_height_new, bbox_width_new

            quadrants = [
                ("top_left",
                 (new_x_min, new_y_min, new_x_min + int(image_width / 3), new_y_min + int(image_height / 3))),
                # ("top", (new_x_min + int(image_width / 3), new_y_min, new_x_min + int(image_width * 2 / 3),
                #          new_y_min + int(image_height / 3))),
                ("top", (new_x_min + int(image_width / 3), new_y_min, new_x_min + int(image_width * 2 / 3),
                         new_y_min + int(image_height * 0.45))),
                ("top_right",
                 (new_x_min + int(image_width * 2 / 3), new_y_min, new_x_max, new_y_min + int(image_height / 3))),
                ("left", (new_x_min, new_y_min + int(image_height / 3), new_x_min + int(image_width / 3),
                          new_y_min + int(image_height * 2 / 3))),
                ("center", (
                    new_x_min + int(image_width / 3), new_y_min + int(image_height / 3),
                    new_x_min + int(image_width * 2 / 3),
                    new_y_min + int(image_height * 0.70))),
                # ("center", (
                #     new_x_min + int(image_width / 3), new_y_min + int(image_height * 0.55),
                #     new_x_min + int(image_width * 2 / 3), new_y_min + int(image_height * 0.45))),
                ("right", (new_x_min + int(image_width * 2 / 3), new_y_min + int(image_height / 3), new_x_max,
                           new_y_min + int(image_height * 2 / 3))),
                (
                    "bottom_left", (
                        new_x_min, new_y_min + int(image_height * 2 / 3), new_x_min + int(image_width / 3), new_y_max)),
                # ("bottom", (
                #     new_x_min + int(image_width / 3), new_y_min + int(image_height * 2 / 3),
                #     new_x_min + int(image_width * 2 / 3),
                #     new_y_max)),
                ("bottom", (
                    new_x_min + int(image_width / 3), new_y_min + int(image_height * 0.70),
                    new_x_min + int(image_width * 2 / 3),
                    new_y_max)),
                ("bottom_right",
                 (new_x_min + int(image_width * 2 / 3), new_y_min + int(image_height * 2 / 3), new_x_max,
                  new_y_max)),
            ]
            # #############
            top_x_min = new_x_min + int(image_width / 3)
            top_y_min = new_y_min
            top_x_max = new_x_min + int(image_width * 2 / 3)
            top_y_max = new_y_min + int(image_height * 0.47)
            # cv2.rectangle(frame, (top_x_min, top_y_min), (top_x_max, top_y_max), (255, 0, 255), 2)


            bottom_x_min = new_x_min + int(image_width / 3)
            bottom_y_min = new_y_min + new_y_min + int(image_height * 0.5)
            bottom_x_max = new_x_min + int(image_width * 2 / 3)
            bottom_y_max = new_y_max
            # cv2.rectangle(frame, (bottom_x_min, bottom_y_min), (bottom_x_max, bottom_y_max), (0, 0, 255), 2)

            center_x_min = new_x_min + int(image_width / 3)
            center_y_min = new_y_min + int(image_height / 3)
            center_x_max = new_x_min + int(image_width * 2 / 3)
            center_y_max = new_y_min + int(image_height * 0.6)
            # cv2.rectangle(frame, (center_x_min, center_y_min), (center_x_max, center_y_max), (0, 255, 255), 2)


            ################
            gaze_direction = "Unknown"
            for quadrant, (x_min, y_min, x_max, y_max) in quadrants:
                print(f"Checking quadrant: {quadrant}, Bounds: {(x_min, y_min, x_max, y_max)}")
                if x_min <= gaze_point[0] <= x_max and y_min <= gaze_point[1] <= y_max:
                    print(f"✅ Gaze detected in quadrant: {quadrant}")
                    quadrant_gaze = f"Gaze: {quadrant}"
                    gaze_direction = quadrant

                    cv2.putText(frame, quadrant_gaze, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    break

            result_dict["gaze"] = gaze_direction

####################### vẽ trung tâm và cảnh báo #################
            # Xác định vùng trung tâm (ô giữa trong 3x3 grid)
            center_x1 = int(img_w * 1 / 3)
            center_y1 = int(img_h * 1 / 3)
            center_x2 = int(img_w * 2 / 3)
            center_y2 = int(img_h * 2 / 3)

            # Vẽ vùng trung tâm để người dùng căn chỉnh
            cv2.rectangle(frame, (center_x1, center_y1), (center_x2, center_y2), (0, 255, 0), 2)

            # Kiểm tra xem bbox có nằm hoàn toàn trong vùng trung tâm không
            # if not (x_min >= center_x1 and y_min >= center_y1 and x_max <= center_x2 and y_max <= center_y2):
            if not (center_x1 <= center_x <= center_x2 and center_y1 <= center_y <= center_y2):
                cv2.putText(
                    frame,
                    "Please center your face!",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )
####################### vẽ trung tâm và cảnh báo #################

####################### Cảnh báo cheating #################

            cheating = f"No cheating"
            len_bboxes = results.bboxes
            if quadrant == "top":
                cheating = f"Cheating"
                cv2.putText(frame, cheating, (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)


            elif (
                    gaze_point_new[0] < 0 or gaze_point_new[0] > screen_width or
                    gaze_point_new[1] < 0 or gaze_point_new[1] > screen_height
            ) and quadrant != "center":
                cheating = "Cheating"
                cv2.putText(frame, cheating, (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)


            elif len(len_bboxes) > 1:
                cheating = f"Cheating"
                cv2.putText(frame, cheating, (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            is_within_screen = (
                    0 <= gaze_point_new[0] <= screen_width and
                    0 <= gaze_point_new[1] <= screen_height
            )

            # Store the result in the dictionary
            result_dict["in_screen"] = is_within_screen

            result_dict["Cheating"] = cheating

            ####################### cảnh báo cheating #################

            ########### --> Face Spoofing
            model_dir = r"D:\PyCharm\pythonProject\Face_Rec\Silent_Face_Anti_Spoofing\resources\anti_spoof_models"
            frame = test(frame.copy(), model_dir=model_dir, device_id=0)
            # result_dict["spoofing"] = spoof_result if spoof_result else "Unknown"

    ##############################################

            # Convert frame to grayscale for face detection
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces in the frame using the cascade classifier
    ############### use haarcascade
            # faces = haar_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
            #
            # # Loop through the detected faces
            # for (x, y, w, h) in faces:
            for bbox in results.bboxes:
                # Chuyển đổi các giá trị bounding box sang kiểu int
                x, y, x_max, y_max = map(int, bbox)
                w = x_max - x
                h = y_max - y
                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 155, 255), 2)


                # Extract the face region from the frame
                face_region = frame[y:y+h, x:x+w]

                # Preprocess the face region for prediction
                X = self.face_preprocessor(face_region, 0, 0, w, h, required_size=(64, 64))
                # Predict the class label and its probability for the face
                label, probability = self.face_svm_classifier(X)
                print("Person: {}, Probability: {}".format(label, probability))
                # Add the detected class label to the frame
                if float(probability) > 0.6 :
                    result_dict["face_label"] = label
                    result_dict["face_probability"] = probability
                    cv2.putText(frame, label+probability, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                else:
                    result_dict["face_label"] = "unknown"
                    result_dict["face_probability"] = probability

            # Display the frame with labels
            # cv2.imshow('frame', frame)
            # Break on keyboard interruption with 'q'
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        else:
            print("Không có kết quả gaze, xử lý tiếp mà không có thông tin gaze.")

        # Đảm bảo FPS không vượt quá target_fps
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_time - elapsed_time)
        time.sleep(sleep_time)



        output_dir = r"D:\PyCharm\pythonProject\Face_Rec\Silent_Face_Anti_Spoofing\images\sample"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "output_frame.jpg")

        # Lưu ảnh
        output_frame = cv2.imwrite(output_path, frame)

        # return result_dict
        return output_frame

#################################
        # while True:
        #     # Capture frame-by-frame
        #     __, frame1 = cap.read()
        #
        #     frame1 = cv2.flip(frame1, 1)
        #
        #     frame = equalize_brightness(frame1)
        #
        #     # 1. Extract faces from frames
        #     result = self.face_mtcnn_extractor(frame)
        #
        #     if result:
        #         for person in result:
        #             # 2. Localize the face in the frame
        #             x1, y1, x2, y2, width, height = self.face_localizer(person)
        #             # 3. Proprocess the images for prediction
        #             X = self.face_preprocessor(frame, x1, y1, x2, y2, required_size=(64, 64))
        #             # 4. Predict class label and its probability
        #             label, probability = self.face_svm_classifier(X)
        #             print(" Person : {} , Probability : {}".format(label, probability))
        #             # 5. Draw a frame
        #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 155, 255), 2)
        #             # 6. Add the detected class label to the frame
        #             cv2.putText(frame, label+probability, (x1, y2+40),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
        #                         lineType=cv2.LINE_AA)
        #     # display the frame with label
        #     cv2.imshow('frame', frame)
        #     # break on keybord interuption with 'q'
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # # When everything's done, release capture
        # cap.release()
        # cv2.destroyAllWindows()

############################################################
        # dectector = FaceDetector()
        # offsetPercentageW = 10
        # offsetPercentageH = 20
        # while True:
        #     # Capture frame-by-frame
        #     __, frame = cap.read()
        #
        #     # frame = equalize_brightness(frame1)
        #     frame = cv2.flip(frame, 1)
        #
        #     # 1. Extract faces from frames
        #     img, bboxs = dectector.findFaces(frame, draw=False)
        #     if bboxs:
        #         for bbox in bboxs:
        #             x, y, w, h = bbox['bbox']
        #             offsetW = (offsetPercentageW / 100) * w
        #             x = int(x - offsetW)
        #             w = int(w + offsetW * 2)
        #
        #             offsetH = (offsetPercentageH / 100) * h
        #             y = int(y - offsetH * 3)
        #             h = int(h + offsetH * 3.5)
        #             #
        #             if x < 0: x = 0
        #             if y < 0: y = 0
        #             if w < 0: w = 0
        #             if h < 0: h = 0
        #             # Draw a rectangle around the detected face
        #             # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 155, 255), 2)
        #             cvzone.cornerRect(img, (x, y, w, h))
        #
        #             # Extract the face region from the frame
        #             face_region = img[y:y + h, x:x + w]
        #
        #             # Preprocess the face region for prediction
        #             X = self.face_preprocessor(face_region, 0, 0, w, h, required_size=(64, 64))
        #             # Predict the class label and its probability for the face
        #             label, probability = self.face_svm_classifier(X)
        #             print("Person: {}, Probability: {}".format(label, probability))
        #             # Add the detected class label to the frame
        #             if float(probability) > 0.6:
        #                 cv2.putText(frame, label + probability, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        #
        #     # Display the frame with labels
        #     cv2.imshow('frame', frame)
        #     # Break on keyboard interruption with 'q'
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        #
        # # Release capture when finished
        # cap.release()
        # cv2.destroyAllWindows()
############################################################



if __name__ == "__main__":
    # image_path = r"D:\PyCharm\pythonProject\Face_Rec\FaceDataset\val\to_van_tu\50.jpg"
    image_path = r"D:\oes\eyetracking\images\anh2.png"

    facedetector = FaceDetectors()


    start_time = time.time()

    result = facedetector.face_detector(image_path)

    end_time = time.time()
    execution_time = end_time - start_time

    # In kết quả và thời gian chạy
    print("Kết quả:", result)
    print(f"Thời gian chạy: {execution_time:.2f} giây")