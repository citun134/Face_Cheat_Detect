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
# from test import test



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

    def face_detector(self):
        """Method classifies faces on live cam feed
           Class labels : sai_ram, donald_trump,narendra_modi, virat_koli"""
        # open cv for live cam feed
        cap = cv2.VideoCapture(0)
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

        while True:
                start_time = time.time()  # Bắt đầu đo thời gian
                # Capture frame-by-frame
                __, frame1 = cap.read()

                frame = cv2.flip(frame1, 1)
                # frame = equalize_brightness(frame)


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
                            new_y_min + int(image_height * 2 / 3))),
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
                            new_x_min + int(image_width / 3), new_y_min + int(image_height * 0.55),
                            new_x_min + int(image_width * 2 / 3),
                            new_y_max)),
                        ("bottom_right",
                         (new_x_min + int(image_width * 2 / 3), new_y_min + int(image_height * 2 / 3), new_x_max,
                          new_y_max)),
                    ]
                    #############
                    bottom_x_min = new_x_min + int(image_width / 3)
                    bottom_y_min = new_y_min + new_y_min + int(image_height * 0.55)
                    bottom_x_max = new_x_min + int(image_width * 2 / 3)
                    bottom_y_max = new_y_max

                    center_x_min = new_x_min + int(image_width / 3)
                    center_y_min = new_y_min + int(image_height * 0.55 )
                    center_x_max = new_x_min + int(image_width * 2 / 3)
                    center_y_max = new_y_min + int(image_height * 0.45)
                    cv2.rectangle(frame, (center_x_min, center_y_min), (center_x_max, center_y_max), (0, 0, 255), 2)


                    cv2.rectangle(frame, (bottom_x_min, bottom_y_min), (bottom_x_max, bottom_y_max), (0, 0, 255), 2)
                    ################

                    for quadrant, (x_min, y_min, x_max, y_max) in quadrants:
                        print(f"Checking quadrant: {quadrant}, Bounds: {(x_min, y_min, x_max, y_max)}")
                        if x_min <= gaze_point[0] <= x_max and y_min <= gaze_point[1] <= y_max:
                            print(f"✅ Gaze detected in quadrant: {quadrant}")
                            quadrant_gaze = f"Gaze: {quadrant}"

                            cv2.putText(frame, quadrant_gaze, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                            break

                    len_bboxes = results.bboxes
                    if text != "Center" and quadrant != "center":
                        cheating = f"Cheating"
                        cv2.putText(frame, cheating, (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    elif quadrant != "center":
                        cheating = f"Cheating"
                        cv2.putText(frame, cheating, (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    elif len(len_bboxes) > 1:
                        cheating = f"Cheating"
                        cv2.putText(frame, cheating, (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            ########### --> Face Spoofing
                    model_dir = r"D:\PyCharm\pythonProject\Face_Rec\Silent_Face_Anti_Spoofing\resources\anti_spoof_models"
                    # test(frame, model_dir, device_id=0)

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
                        if float(probability) > 0.6:
                            cv2.putText(frame, label+probability, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                    # Display the frame with labels
                    cv2.imshow('frame', frame)
                    # Break on keyboard interruption with 'q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                else:
                    print("Không có kết quả gaze, xử lý tiếp mà không có thông tin gaze.")

                # Đảm bảo FPS không vượt quá target_fps
                elapsed_time = time.time() - start_time
                sleep_time = max(0, frame_time - elapsed_time)
                time.sleep(sleep_time)

        # Release capture when finished
        cap.release()
        cv2.destroyAllWindows()


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
    facedetector = FaceDetectors()
    facedetector.face_detector()
