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

from test import test

import argparse

from src.anti_spoof_predict import AntiSpoofPredict
# from Silent_Face_Anti_Spoofing.src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

from screeninfo import get_monitors


class FaceDetectors:

    def __init__(self):
        self.facenet_model = FaceNet()
        self.svm_model = pickle.load(open("D:/PyCharm/pythonProject/Face_Rec/FaceDataset/SVM_classifier.sav", 'rb'))
        self.data = np.load('D:/PyCharm/pythonProject/Face_Rec/FaceDataset/faces_dataset_embeddings.npz')
        # object to the MTCNN detector class
        self.detector = MTCNN()

################# START CODE FACE RECOGNITION ############################

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

### ################# END CODE FACE RECOGNITION ############################

### ################# START CODE HEAD POSE ############################
    def extract_features(self, img, face_mesh):
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

    def normalize(self, poses_df):
        normalized_df = poses_df.copy()

        for dim in ['x', 'y']:
            # Centerning around the nose
            for feature in ['forehead_' + dim, 'nose_' + dim, 'mouth_left_' + dim, 'mouth_right_' + dim,
                            'left_eye_' + dim,
                            'chin_' + dim, 'right_eye_' + dim]:
                normalized_df[feature] = poses_df[feature] - poses_df['nose_' + dim]

            # Scaling
            diff = normalized_df['mouth_right_' + dim] - normalized_df['left_eye_' + dim]
            for feature in ['forehead_' + dim, 'nose_' + dim, 'mouth_left_' + dim, 'mouth_right_' + dim,
                            'left_eye_' + dim,
                            'chin_' + dim, 'right_eye_' + dim]:
                normalized_df[feature] = normalized_df[feature] / diff

        return normalized_df

    def draw_axes(self, img, pitch, yaw, roll, tx, ty, size=50):
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

### ################# END CODE HEAD POSE ############################

### ################# START CODE EQUALIZE BRIGHTNESS ############################

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

### ################# END CODE EQUALIZE BRIGHTNESS ############################

### ################# START CODE lệch camera ############################

    def gaze_vector_from_angles(self, pitch, yaw):
        # pitch, yaw tính bằng radian
        x = np.cos(pitch) * np.sin(yaw)
        y = -np.sin(pitch)
        z = np.cos(pitch) * np.cos(yaw)
        return np.array([x, y, z])

    def rotation_matrix_yaw(self, angle_deg):
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)  # Dấu trừ vì muốn xoay ngược
        sin_a = np.sin(-angle_rad)
        return np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])

    def correct_gaze_for_camera_angle(self, pitch, yaw, cam_yaw_offset_deg=30):
        gaze_vec = self.gaze_vector_from_angles(pitch, yaw)
        R = self.rotation_matrix_yaw(cam_yaw_offset_deg)
        corrected_gaze_vec = R @ gaze_vec

        # Convert corrected vector back to pitch & yaw
        x, y, z = corrected_gaze_vec
        corrected_pitch = -np.arcsin(y)
        corrected_yaw = np.arctan2(x, z)
        return corrected_pitch, corrected_yaw

    def calculate_gaze_point(self, pitch, yaw, screen_width_px, screen_height_px, width_cm, height_cm, lrate=1.0,
                             cam_yaw_offset_deg=30):
        # Bước 1: Sửa lại góc nhìn theo góc camera
        pitch, yaw = self.correct_gaze_for_camera_angle(pitch, yaw, cam_yaw_offset_deg)

        # Bước 2: Tính tỉ lệ cm/pixel
        cm_per_pixel_x = width_cm / screen_width_px
        cm_per_pixel_y = height_cm / screen_height_px

        # Bước 3: Tính tọa độ điểm nhìn
        dx_cm = -width_cm * np.sin(pitch) * np.cos(yaw) * lrate
        dy_cm = -height_cm * np.sin(yaw) * lrate

        dx_px = dx_cm / cm_per_pixel_x
        dy_px = dy_cm / cm_per_pixel_y

        return dx_px, dy_px

    ### ################# START CODE lệch camera ############################

### ################# START CODE MAIN ############################

    def face_detector(self, image_path):
        """Method classifies faces on live cam feed
           Class labels : sai_ram, donald_trump,narendra_modi, virat_koli"""

        frame = cv2.imread(image_path)  ## --> nếu sử dụng lưu ảnh tạm trong thư mục
        # frame = image_path ## --> nếu sử dụng đọc ảnh từ api

        frame_height, frame_width = frame.shape[:2]
        frame_height, frame_width = frame.shape[:2]

        print("Chiều rộng:", frame_width)
        print("Chiều cao:", frame_height)

        if frame is None:
            return {"error": f"Không thể đọc ảnh từ {image_path}"}

####### ################# START CODE ############################

        # model head pose
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

        # Xử dụng haarcascade để detect khuôn mặt
        haar_cascade = cv2.CascadeClassifier('D:/PyCharm/pythonProject/Face_Rec/haarcascade_frontalface_default.xml')

####### ############## bật webcam ################
        start_time = time.time()  # Bắt đầu đo thời gian
        # Capture frame-by-frame

        frame = cv2.flip(frame, 1)
        # frame = equalize_brightness(frame)
        frame = cv2.resize(frame, (frame_width, frame_height))
        img_h, img_w, img_c = frame.shape

        result_dict = {}

############### ######### START CODE HEAD POSE ###############

        cols = []
        for pos in ['nose_', 'forehead_', 'left_eye_', 'mouth_left_', 'chin_', 'right_eye_', 'mouth_right_']:
            for dim in ('x', 'y'):
                cols.append(pos + dim)

        text = ''

        # Giả định các hàm extract_features, normalize, draw_axes đã được định nghĩa
        face_features = self.extract_features(frame, face_mesh)
        if len(face_features):
            face_features_df = pd.DataFrame([face_features], columns=cols)
            face_features_normalized = self.normalize(face_features_df)
            pitch_pred, yaw_pred, roll_pred = model.predict(face_features_normalized).ravel()
            print(f"pitch_pred: {pitch_pred}, yaw_pred: {yaw_pred}, roll_pred: {roll_pred}")

            nose_x = face_features_df['nose_x'].values * img_w
            nose_y = face_features_df['nose_y'].values * img_h
            print(f"nose_x: {nose_x}, nose_y: {nose_y}")
            frame = self.draw_axes(frame, pitch_pred, yaw_pred, roll_pred, nose_x, nose_y)

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
        # result_dict["head_pose"] = text

        print(head_pose_text)
        # Vẽ text lên ảnh
        cv2.putText(frame, head_pose_text, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

############### ######### END CODE HEAD POSE ###############

        # Chuyển lại sang BGR để hiển thị và lưu
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Xử lý ảnh với mô hình Gaze Estimation
        # results = gaze_pipeline.step(frame)

############### ######### START CODE GAZE ESTIMATION ###############

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

            # Lấy thông tin từ bounding box của mặt
            x_min, y_min, x_max, y_max = bbox
            # Tính toán chiều rộng và chiều cao của bounding box
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            ################### ######### START tính khoảng cách mặt ###############
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

            result_dict["distance"] = float(f"{distance:.2f}")


            # if distance:
            #     cv2.putText(frame, f"{distance:.2f} cm", (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            #     print(f"Khoảng cách: {distance:.2f} cm")
            # else:
            #     print("Không thể tính khoảng cách")

################### ######### END tính khoảng cách mặt ###############

################### ######### START mắt đang nhìn đâu trên màn hình point xanh ###############
            # Tính toán tọa độ trung tâm của bounding box
            center_x = int(x_min + bbox_width / 2)
            center_y = int(y_min + bbox_height / 2)

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
                lrate = 1.45
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
                print("Khoảng cách hợp lý  < 35cm")
                lrate = 1.0
                dx_new = -screen_width * np.sin(pitch) * np.cos(yaw) * lrate  # Khoảng cách ngang từ trung tâm
                dy_new = -screen_height * np.sin(yaw) * lrate
                gaze_point_new = (int(center_x + dx_new), int(center_y + dy_new))

            # cv2.circle(frame, gaze_point_new, 15, (0, 255, 0), -1)
            #
            # corner_size = 0.25
            # corner_width = int(screen_width * corner_size)
            # corner_height = int(screen_height * corner_size)
            # corners = {
            #     "top_left": (0, 0, corner_width, corner_height),
            #     "top_right": (screen_width - corner_width, 0, screen_width, corner_height),
            #     "bottom_left": (0, screen_height - corner_height, corner_width, screen_height),
            #     "bottom_right": (
            #         screen_width - corner_width, screen_height - corner_height, screen_width, screen_height)
            # }
            #
            # corner_detected = "none"
            # colors = {
            #     "top_left": (0, 255, 0),  # Xanh lá
            #     "top_right": (0, 0, 255),  # Đỏ
            #     "bottom_left": (255, 0, 0),  # Xanh dương
            #     "bottom_right": (0, 255, 255)  # Vàng
            # }
            #
            # for corner_name, (x_min, y_min, x_max, y_max) in corners.items():
            #     # Vẽ hình chữ nhật cho mỗi góc
            #     cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), colors[corner_name], 2)
            #     # Thêm nhãn tên góc
            #     cv2.putText(frame, corner_name, (x_min + 5, y_min + 20),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[corner_name], 2)

################### ######### END mắt đang nhìn đâu trên màn hình point xanh ###############

################### ######### START mắt đang nhìn đâu trên màn hình point đỏ ###############

            # Tính toán tọa độ trung tâm của bounding box
            center_x = int(x_min + bbox_width / 2)
            center_y = int(y_min + bbox_height / 2)

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

            # Tính point đỏ cho mắt, xác định hướng
            # Tính toán tọa độ điểm gaze dựa vào công thức mới
            dx = -bbox_width * np.sin(pitch) * np.cos(yaw)  # Sử dụng công thức mới cho dx
            dy = -bbox_width * np.sin(yaw)  # Sử dụng công thức mới cho dy

            # Tính toán tọa độ điểm gaze trong bounding box
            gaze_point = (int(center_x + dx), int(center_y + dy))

            # Vẽ điểm gaze trên ảnh
            # cv2.circle(frame, gaze_point, 10, (0, 0, 255), -1)

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
                        new_x_min, new_y_min + int(image_height * 2 / 3), new_x_min + int(image_width / 3),
                        new_y_max)),
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

################### ######### START vẽ phạm vi nhận hướng của mắt ###############

            # top_x_min = new_x_min + int(image_width / 3)
            # top_y_min = new_y_min
            # top_x_max = new_x_min + int(image_width * 2 / 3)
            # top_y_max = new_y_min + int(image_height * 0.45)
            # cv2.rectangle(frame, (top_x_min, top_y_min), (top_x_max, top_y_max), (255, 0, 255), 2)
            #
            # bottom_x_min = new_x_min + int(image_width / 3)
            # bottom_y_min = new_y_min + new_y_min + int(image_height * 0.70)
            # bottom_x_max = new_x_min + int(image_width * 2 / 3)
            # bottom_y_max = new_y_max
            # cv2.rectangle(frame, (bottom_x_min, bottom_y_min), (bottom_x_max, bottom_y_max), (0, 0, 255), 2)
            #
            # center_x_min = new_x_min + int(image_width / 3)
            # center_y_min = new_y_min + int(image_height / 3)
            # center_x_max = new_x_min + int(image_width * 2 / 3)
            # center_y_max = new_y_min + int(image_height * 0.70)
            # cv2.rectangle(frame, (center_x_min, center_y_min), (center_x_max, center_y_max), (0, 255, 255), 2)

################### ######### END vẽ phạm vi nhận hướng của mắt ###############
            gaze_direction = "center"

            for quadrant, (x_min, y_min, x_max, y_max) in quadrants:
                print(f"Checking quadrant: {quadrant}, Bounds: {(x_min, y_min, x_max, y_max)}")
                if x_min <= gaze_point[0] <= x_max and y_min <= gaze_point[1] <= y_max:
                    print(f"✅ Gaze detected in quadrant: {quadrant}")
                    quadrant_gaze = f"Gaze: {quadrant}"
                    gaze_direction = quadrant

                    # cv2.putText(frame, quadrant_gaze, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    break

            result_dict["gaze"] = gaze_direction

################### ######### END mắt đang nhìn đâu trên màn hình point đỏ ###############

################### ######### START vẽ trung tâm và cảnh báo #################
            # # Xác định vùng trung tâm (ô giữa trong 3x3 grid)
            # center_x1 = int(img_w * 1 / 3)
            # center_y1 = int(img_h * 1 / 3)
            # center_x2 = int(img_w * 2 / 3)
            # center_y2 = int(img_h * 2 / 3)
            #
            # # Vẽ vùng trung tâm để người dùng căn chỉnh
            # cv2.rectangle(frame, (center_x1, center_y1), (center_x2, center_y2), (0, 255, 0), 2)
            #
            # # Kiểm tra xem bbox có nằm hoàn toàn trong vùng trung tâm không
            # # if not (x_min >= center_x1 and y_min >= center_y1 and x_max <= center_x2 and y_max <= center_y2):
            # if not (center_x1 <= center_x <= center_x2 and center_y1 <= center_y <= center_y2):
            #     cv2.putText(
            #         frame,
            #         "Please center your face!",
            #         (30, 80),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         1,
            #         (0, 0, 255),
            #         3
            #     )

################### ######### END vẽ trung tâm và cảnh báo #################

################### ######### START Cảnh báo cheating #################

            len_bboxes = results.bboxes
            cheating = f"No cheating"

            # if quadrant == "top" or quadrant == "bottom":
            #    cheating = f"Cheating"
            #    cv2.putText(frame, cheating, (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            #
            # if (
            #         0 <= gaze_point_new[0] <= screen_width and
            #         0 <= gaze_point_new[1] <= screen_height
            # ):
            #     is_within_screen = True
            #     # Không có cheating nếu trong màn hình
            # else:
            #     is_within_screen = False
            #
            #     # Chỉ gán là Cheating nếu hướng nhìn lệch sang trái/phải và ra khỏi màn hình
            #     if quadrant in ["right", "left"] and quadrant != "center":
            #         cheating = "Cheating"
            #         cv2.putText(frame, cheating, (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            #     else:
            #         cheating = "No Cheating"
            # Mặc định
            cheating = "No Cheating"

            # Trường hợp 1: Nhìn lên hoặc xuống là luôn cheating
            if quadrant in ["top", "bottom"]:
                cheating = "Cheating"
                cv2.putText(frame, cheating, (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            # Kiểm tra xem ánh nhìn có trong màn hình không
            if (
                    0 <= gaze_point_new[0] <= screen_width and
                    0 <= gaze_point_new[1] <= screen_height
            ):
                is_within_screen = True
            else:
                is_within_screen = False

                # Trường hợp 2: Ra khỏi màn hình và nhìn trái/phải → cheating
                if quadrant in ["right", "left"]:
                    cheating = "Cheating"
                    cv2.putText(frame, cheating, (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            # Trường hợp 3: Nhìn chính diện (center) thì luôn là No Cheating
            if quadrant == "center":
                cheating = "No Cheating"
            # elif (
            #         gaze_point_new[0] < 0 or gaze_point_new[0] > screen_width or
            #         gaze_point_new[1] < 0 or gaze_point_new[1] > screen_height
            # ) and quadrant == "right":
            #     cheating = "Cheating"
            #     cv2.putText(frame, cheating, (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            #
            # elif (
            #         gaze_point_new[0] < 0 or gaze_point_new[0] > screen_width or
            #         gaze_point_new[1] < 0 or gaze_point_new[1] > screen_height
            # ) and quadrant == "left":
            #     cheating = "Cheating"
            #     cv2.putText(frame, cheating, (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)


            if len(len_bboxes) > 1:
                cheating = f"Cheating"
                cv2.putText(frame, cheating, (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)


            is_within_screen = (
                    0 <= gaze_point_new[0] <= screen_width and
                    0 <= gaze_point_new[1] <= screen_height
            )

            result_dict["in_screen"] = is_within_screen

            result_dict["person"] = len(len_bboxes)

            result_dict["Cheating"] = cheating
################### ######### END Cảnh báo cheating #################

################### ######### START code face anti spoofing #################

            # model_dir = r"D:\PyCharm\pythonProject\Face_Rec\Silent_Face_Anti_Spoofing\resources\anti_spoof_models"
            # frame = test(frame.copy(), model_dir=model_dir, device_id=0)

################### ######### END code face anti spoofing #################

################### ######### START code face recognition #################

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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 155, 255), 2)

                # Extract the face region from the frame
                face_region = frame[y:y + h, x:x + w]

                # Preprocess the face region for prediction
                X = self.face_preprocessor(face_region, 0, 0, w, h, required_size=(64, 64))
                # Predict the class label and its probability for the face
                label, probability = self.face_svm_classifier(X)
                result_dict["face_label"] = label
                # result_dict["face_probability"] = probability
                result_dict["face_probability"] = float(f"{float(probability):.2f}")


                print("Person: {}, Probability: {}".format(label, probability))


                # # Add the detected class label to the frame
                # if float(probability) > 0.6:
                #     # result_dict["face_label"] = label
                #     result_dict["face_probability"] = probability
                #     cv2.putText(frame, label + probability, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                #                 2)
                # else:
                #     # result_dict["face_label"] = "unknown"
                #     result_dict["face_probability"] = probability

            ################### ######### END code face recognition #################

        else:
            print("Không có kết quả gaze, xử lý tiếp mà không có thông tin gaze.")
        ############### ######### END CODE GAZE ESTIMATION ###############

        # Đảm bảo FPS không vượt quá target_fps
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_time - elapsed_time)
        time.sleep(sleep_time)

        # output_dir = r"D:\PyCharm\pythonProject\Face_Rec\Silent_Face_Anti_Spoofing\images\sample"
        # os.makedirs(output_dir, exist_ok=True)
        # output_path = os.path.join(output_dir, "output_frame.jpg")
        #
        # # Lưu ảnh
        # output_frame = cv2.imwrite(output_path, frame)
        # return output_frame

        return result_dict


### ################# END CODE MAIN ############################


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

################# START CODE MAIN ############################


if __name__ == "__main__":
    # image_path = r"D:\PyCharm\pythonProject\Face_Rec\FaceDataset\val\to_van_tu\50.jpg"
    image_path = r"D:\oes\eyetracking\images\anh1.png"

    facedetector = FaceDetectors()


    start_time = time.time()

    result = facedetector.face_detector(image_path)

    end_time = time.time()
    execution_time = end_time - start_time

    # In kết quả và thời gian chạy
    print("Kết quả:", result)
    print(f"Thời gian chạy: {execution_time:.2f} giây")