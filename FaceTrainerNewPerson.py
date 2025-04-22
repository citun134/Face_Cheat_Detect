################################
# Face detection Trainer       #
################################

# import libraries
import warnings
warnings.filterwarnings("ignore")
import datetime
import time
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed, asarray, load, expand_dims
from mtcnn.mtcnn import MTCNN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from keras_facenet import FaceNet
import pickle

from keras.src.legacy.preprocessing.image import ImageDataGenerator
import numpy as np
import os

class FaceTrainer:

    def __init__(self):
        self.dataset_train = "D:/PyCharm/pythonProject/Face_Rec/FaceDataset/train/"
        self.dataset_val = "D:/PyCharm/pythonProject/Face_Rec/FaceDataset/val/"
        self.faces_npz = "D:/PyCharm/pythonProject/Face_Rec/FaceDataset/faces_dataset.npz"
        self.keras_facenet =  FaceNet()
        self.faces_embeddings = "D:/PyCharm/pythonProject/Face_Rec/FaceDataset/faces_dataset_embeddings.npz"
        self.svm_classifier = "D:/PyCharm/pythonProject/Face_Rec/FaceDataset/SVM_classifier.sav"

        self.labelstxt = "D:/PyCharm/pythonProject/Face_Rec/labels.txt"
        return

    # def load_dataset(self, directory):
    #     """Load a dataset that contains one subdir for each class that in turn contains images"""
    #     X = []
    #     y = []
    #     # enumerate all folders named with class labels
    #     for subdir in listdir(directory):
    #         path = directory + subdir + '/'
    #         # skip any files that might be in the dir
    #         if not isdir(path):
    #             continue
    #         # load all faces in the subdirectory
    #         faces = self.load_faces(path)
    #         # create labels
    #         labels = [subdir for _ in range(len(faces))]
    #         print("loaded {} examples for class: {}".format(len(faces), subdir))
    #         X.extend(faces)
    #         y.extend(labels)
    #     return asarray(X), asarray(y)

    ##################
    def save_subdirs_to_txt(self, directory, txt_file):
        """Save subdirectories to a text file"""
        with open(txt_file, 'w') as f:
            for subdir in os.listdir(directory):
                if os.path.isdir(os.path.join(directory, subdir)):
                    f.write(subdir + '\n')

    def check_and_run_new_subdirs(self, directory, txt_file):
        """Check for new subdirectories in the directory and run code for each new subdir"""
        with open(txt_file, 'r') as f:
            existing_subdirs = set(f.read().splitlines())

        new_subdirs = []
        for subdir in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, subdir)) and subdir not in existing_subdirs:
                new_subdirs.append(subdir)

        # Check if there are any new subdirectories found
        if new_subdirs:
            # Run code for each new subdir
            for new_subdir in new_subdirs:
                # subdir_path = os.path.join(directory, new_subdir)
                # Run code for subdir_path
                self.load_dataset(directory, new_subdir)
                # Add more code as needed
            with open(txt_file, 'a') as f:
                for new_subdir in new_subdirs:
                    f.write(new_subdir + '\n')

        # Return a default value in case no new subdirectories were found
        return (), ()


    ##################


    ###############
    def load_dataset(self, directory, txt_file):
        """Load a dataset that contains one subdir for each class that in turn contains images"""
        X = []
        y = []
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )

        with open(txt_file, 'r') as f:
            existing_subdirs = set(f.read().splitlines())

        new_subdirs = []
        for subdir in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, subdir)) and subdir not in existing_subdirs:
                new_subdirs.append(subdir)

        if new_subdirs:
            # Run code for each new subdir
            for subdir in new_subdirs:

                # current_subdirs_txt = "labels.txt"
                # self.save_subdirs_to_txt(directory, current_subdirs_txt)

                # for subdir in listdir(directory):
                path = directory + subdir + '/'
                # if not isdir(path):
                #     continue

                faces = self.load_faces(path)
                labels = [subdir for _ in range(len(faces))]

                if directory == self.dataset_val:
                    with open(txt_file, 'a') as f:
                        f.write(subdir + '\n')

                for i, face in enumerate(faces):
                    X.append(face)
                    y.append(labels[i])

                    # Áp dụng biến đổi dựa trên datagen cho từng ảnh và thêm vào dữ liệu augmentation
                    no_img = 0
                    for x in datagen.flow(expand_dims(face, axis=0), batch_size=1):
                        X.append(x[0])
                        y.append(labels[i])
                        no_img += 1
                        if no_img == 4:  # Số lượng ảnh được tạo ra từ mỗi ảnh gốc
                            break

        return asarray(X), asarray(y)

    ###############

    def load_faces(self, directory):
        """Load images and extract faces for all images in a directory"""
        faces = []
        # enumerate files
        for filename in listdir(directory):
            path = directory + filename
            # get face
            face = self.extract_face(path)
            faces.append(face)
        return faces

    def extract_face(self, filename, required_size=(160, 160)):
        """Extract a single face from a given photograph"""
        # load image from file
        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = asarray(image)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box from the first face

        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array

    # def create_faces_npz(self):
    #     """Method Creates npz file for all the faces in train_dir, val_dir"""
    #     # Load the training data set
    #     # trainX, trainy = self.load_dataset(self.dataset_train)
    #     trainX, trainy = self.check_and_run_new_subdirs(self.dataset_train, self.labelstxt)
    #     print("Training data set loaded")
    #     # load test dataset
    #     # testX, testy = self.load_dataset(self.dataset_val)
    #     testX, testy = self.check_and_run_new_subdirs(self.dataset_val, self.labelstxt)
    #
    #     print("Testing data set loaded")
    #     # save arrays to one file in compressed format
    #     # savez_compressed(self.faces_npz, trainX, trainy, testX, testy)
    #     # Load existing NPZ file
    #     existing_data = np.load(self.faces_npz)
    #     existing_trainX, existing_trainy = existing_data['arr_0'], existing_data['arr_1']
    #     existing_testX, existing_testy = existing_data['arr_2'], existing_data['arr_3']
    #
    #     # Concatenate new data with existing data
    #     updated_trainX = np.concatenate((existing_trainX, trainX), axis=0)
    #     updated_trainy = np.concatenate((existing_trainy, trainy), axis=0)
    #     updated_testX = np.concatenate((existing_testX, testX), axis=0)
    #     updated_testy = np.concatenate((existing_testy, testy), axis=0)
    #
    #     # Save arrays to existing file in compressed format
    #     np.savez_compressed(self.faces_npz, updated_trainX, updated_trainy, updated_testX, updated_testy)
    #     return
    def create_faces_npz(self):
        """Method Creates npz file for all the faces in train_dir, val_dir"""
        # Load the training data set
        trainX, trainy = self.load_dataset(self.dataset_train, self.labelstxt)
        print("Training data set loaded")
        # Load test dataset
        testX, testy = self.load_dataset(self.dataset_val, self.labelstxt)
        print("Testing data set loaded")
        existing_data = np.load(self.faces_npz)
        existing_trainX, existing_trainy = existing_data['arr_0'], existing_data['arr_1']
        existing_testX, existing_testy = existing_data['arr_2'], existing_data['arr_3']

        # Reshape existing labels to match the shape of new labels
        existing_trainy = existing_trainy.reshape(-1, 1)

        # Reshape trainy to match the shape of existing_trainy
        trainy = trainy.reshape(-1, 1)

        # Reshape existing test data to have the same number of dimensions as new test data
        existing_testX = np.squeeze(existing_testX)  # Remove extra dimensions
        existing_testy = existing_testy.reshape(-1, 1)

        # Reshape existing train data to have the same number of dimensions as new train data
        existing_trainX = np.squeeze(existing_trainX)  # Remove extra dimensions

        # Check if the dimensions of existing_trainX and trainX are compatible
        if len(existing_trainX.shape) != len(trainX.shape):
            print("Warning: Dimension mismatch - existing_trainX and trainX have different number of dimensions. "
                  "Concatenating arrays with different dimensions.")

        # Concatenate new data with existing data
        updated_trainX = np.concatenate((existing_trainX, trainX), axis=0)
        updated_trainy = np.concatenate((existing_trainy, trainy), axis=0)
        updated_testX = np.concatenate((existing_testX, testX), axis=0)
        updated_testy = np.concatenate((existing_testy, testy), axis=0)

        # Save arrays to existing file in compressed format
        np.savez_compressed(self.faces_npz, updated_trainX, updated_trainy, updated_testX, updated_testy)

        return

    def create_faces_embedding_npz(self):
        """Create npz file for all the face embeddings in train_dir, val_dir"""
        # Check if data is loaded successfully
        trainX, trainy = self.load_dataset(self.dataset_train, self.labelstxt)
        testX, testy = self.load_dataset(self.dataset_val, self.labelstxt)

        if trainX is not None and testX is not None:
            print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
            # Load the facenet model
            model = self.keras_facenet
            print('Keras Facenet Model Loaded')
            # Convert each face in the train set to an embedding
            newTrainX = []
            for face_pixels in trainX:
                embedding = self.get_embedding(model, face_pixels)
                newTrainX.append(embedding)
            newTrainX = np.asarray(newTrainX)
            # Convert each face in the test set to an embedding
            newTestX = []
            for face_pixels in testX:
                embedding = self.get_embedding(model, face_pixels)
                newTestX.append(embedding)
            newTestX = np.asarray(newTestX)

            # Check if any embeddings were generated for the train set
            if newTrainX.shape[0] == 0:
                print("No embeddings were generated for the train set. Skipping concatenation.")
                return

            # Save arrays to existing file in compressed format
            existing_data = np.load(self.faces_embeddings)
            existing_trainX, existing_trainy = existing_data['arr_0'], existing_data['arr_1']
            existing_testX, existing_testy = existing_data['arr_2'], existing_data['arr_3']

            # Concatenate new labels with existing labels
            updated_trainy = np.concatenate((existing_trainy, trainy), axis=0)
            updated_testy = np.concatenate((existing_testy, testy), axis=0)

            # Create new arrays with correct shapes
            updated_trainX = np.zeros((existing_trainX.shape[0] + newTrainX.shape[0], existing_trainX.shape[1]))
            updated_testX = np.zeros((existing_testX.shape[0] + newTestX.shape[0], existing_testX.shape[1]))

            # Copy data into new arrays
            updated_trainX[:existing_trainX.shape[0], :] = existing_trainX
            updated_trainX[existing_trainX.shape[0]:, :] = newTrainX

            if newTestX.shape[0] > 0:
                updated_testX[:existing_testX.shape[0], :] = existing_testX
                updated_testX[existing_testX.shape[0]:, :] = newTestX
            else:
                updated_testX = existing_testX

            # Save arrays to existing file in compressed format
            np.savez_compressed(self.faces_embeddings, updated_trainX, updated_trainy, updated_testX, updated_testy)
        return

    # def create_faces_embedding_npz(self):
    #     """Create npz file for all the face embeddings in train_dir, val_dir"""
    #     # Check if data is loaded successfully
    #     trainX, trainy = self.load_dataset(self.dataset_train, self.labelstxt)
    #     testX, testy = self.load_dataset(self.dataset_val, self.labelstxt)
    #
    #     if trainX is not None and testX is not None:
    #         print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
    #         # Load the facenet model
    #         model = self.keras_facenet
    #         print('Keras Facenet Model Loaded')
    #         # Convert each face in the train set to an embedding
    #         newTrainX = []
    #         for face_pixels in trainX:
    #             embedding = self.get_embedding(model, face_pixels)
    #             newTrainX.append(embedding)
    #         newTrainX = np.asarray(newTrainX)
    #         # Convert each face in the test set to an embedding
    #         newTestX = []
    #         for face_pixels in testX:
    #             embedding = self.get_embedding(model, face_pixels)
    #             newTestX.append(embedding)
    #         newTestX = np.asarray(newTestX)
    #
    #         # Check if any embeddings were generated for the train set
    #         if newTrainX.shape[0] == 0:
    #             print("No embeddings were generated for the train set. Skipping concatenation.")
    #             return
    #
    #         # Save arrays to existing file in compressed format
    #         existing_data = np.load(self.faces_embeddings)
    #         existing_trainX, existing_trainy = existing_data['arr_0'], existing_data['arr_1']
    #         existing_testX, existing_testy = existing_data['arr_2'], existing_data['arr_3']
    #
    #         # Create new arrays with correct shapes
    #         updated_trainX = np.zeros((existing_trainX.shape[0] + newTrainX.shape[0], existing_trainX.shape[1]))
    #         updated_trainy = np.zeros(existing_trainy.shape[0] + trainy.shape[0],
    #                                   dtype=trainy.dtype)  # Use the dtype of trainy
    #         updated_testX = np.zeros((existing_testX.shape[0] + newTestX.shape[0], existing_testX.shape[1]))
    #         updated_testy = np.zeros(existing_testy.shape[0] + testy.shape[0],
    #                                  dtype=testy.dtype)  # Use the dtype of testy
    #
    #         # Copy data into new arrays
    #         updated_trainX[:existing_trainX.shape[0], :] = existing_trainX
    #         updated_trainX[existing_trainX.shape[0]:, :] = newTrainX
    #
    #         updated_trainy[:existing_trainy.shape[0]] = existing_trainy
    #         updated_trainy[existing_trainy.shape[0]:] = trainy
    #
    #         # Check if newTestX is non-empty before concatenating
    #         if newTestX.shape[0] > 0:
    #             updated_testX[:existing_testX.shape[0], :] = existing_testX
    #             updated_testX[existing_testX.shape[0]:, :] = newTestX
    #
    #             updated_testy[:existing_testy.shape[0]] = existing_testy
    #             updated_testy[existing_testy.shape[0]:] = testy
    #         else:
    #             updated_testX = existing_testX
    #             updated_testy = existing_testy
    #
    #         # Save arrays to existing file in compressed format
    #         np.savez_compressed(self.faces_embeddings, updated_trainX, updated_trainy, updated_testX, updated_testy)
    #     return

    def get_embedding(self, model, face_pixels):
        """Calculate a face embedding for each face in the dataset using facenet
           Get the face embedding for one face"""
        # scale pixel values

        # face_pixels = face_pixels.astype('float32')
        # # standardize pixel values across channels (global)
        # mean, std = face_pixels.mean(), face_pixels.std()
        # face_pixels = (face_pixels - mean) / std

        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding

        # yhat = model.predict(samples)

        yhat = model.embeddings(samples)
        return yhat[0]

    def classifier(self):
        """Create a Classifier for the Faces Dataset"""
        # load dataset
        data = load(self.faces_embeddings)
        trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)
        testX = in_encoder.transform(testX)
        # label encode targets
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        trainy = out_encoder.transform(trainy)
        testy = out_encoder.transform(testy)
        # fit model
        model = SVC(kernel='linear', probability=True)
        model.fit(trainX, trainy)
        # save the model to disk
        filename = self.svm_classifier
        if os.path.exists(filename):
            print("Clearing existing file...")
            with open(filename, 'w') as f:
                f.write('')
            # Dump model to file
        print("Writing new model to file...")
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        # predict
        yhat_train = model.predict(trainX)
        yhat_test = model.predict(testX)
        # score
        score_train = accuracy_score(trainy, yhat_train)
        score_test = accuracy_score(testy, yhat_test)
        # summarize
        print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))
        return

    def clear_file(filename):
        """Clear the content of a file"""
        with open(filename, 'w') as f:
            f.write('')

    def start(self):
        """Method begins the training process"""
        start_time = time.time()
        st = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        print("-----------------------------------------------------------------------------------------------")
        print("Face trainer Initiated at {}".format(st))
        print("-----------------------------------------------------------------------------------------------")
        # Get faces from the images
        # self.create_faces_npz()
        # Get embeddings for all the extracted faces
        self.create_faces_embedding_npz()
        # Classify the faces
        self.classifier()
        end_time = time.time()
        et = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        print("-----------------------------------------------------------------------------------------------")
        print("Face trainer Completed at {}".format(et))
        print("Total time Elapsed {} secs".format(round(end_time - start_time), 0))
        print("-----------------------------------------------------------------------------------------------")

        return


if __name__ == "__main__":
    facetrainer = FaceTrainer()
    facetrainer.start()
