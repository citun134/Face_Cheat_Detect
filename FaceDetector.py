# #############################################
# face detection with mtcnn on live cam feed  #
###############################################
import warnings
warnings.filterwarnings("ignore")
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import cv2
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
# from keras.models import load_model
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer, LabelEncoder
import pickle
import cvzone
from cvzone.FaceDetectionModule import FaceDetector


from imutils.video import FPS


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

        haar_cascade = cv2.CascadeClassifier('D:/PyCharm/pythonProject/Face_Rec/haarcascade_frontalface_default.xml')

        while True:
                # Capture frame-by-frame
                __, frame1 = cap.read()

                frame1 = cv2.flip(frame1, 1)
                frame = equalize_brightness(frame1)

                # Convert frame to grayscale for face detection
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Detect faces in the frame using the cascade classifier
                faces = haar_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

                # Loop through the detected faces
                for (x, y, w, h) in faces:
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
                    cv2.putText(frame, label+probability, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                # Display the frame with labels
                cv2.imshow('frame', frame)
                # Break on keyboard interruption with 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

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
