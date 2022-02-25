import face_recognition
import cv2
import os
import glob
import numpy as np
import mediapipe as mp

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.tolerance = 0.6
        # Resize frame for a faster speed
        self.frame_resizing = 0.3

    def load_data(self):
        """
        load the data from the data files to this object by appending to list of names and faces encodings
        """
        images_path = glob.glob(os.path.join('images/', "**/*.npy"))
        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:

            # Get the name only from the initial file path.

            filename = img_path.split('\\')[1]
            # Get encoding

            with open(img_path, 'rb') as f:
                img_encoding = np.load(f)

            # Store file name and file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def save_encoding_images(self, images_path):
        """
        save encoded images to files for fast loading
        :param images_path:
        """
        # Load Images
        images_path = [fn for fn in glob.glob(os.path.join(images_path, "**/*.*")) if not '.npy' in os.path.basename(fn)]

        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            filepath = img_path.split('.')[0]
            # Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]
            #save the image data to file
            with open(filepath + 'Data.npy', 'wb') as f:
                np.save(f, img_encoding)

        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)

        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
           # matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)

            # get the distances between faces from data to face from the video
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)  # best match found
            if face_distances[best_match_index] <= self.tolerance:
                name = self.known_face_names[best_match_index]  # get the name of this match
                p = (1-face_distances[best_match_index])*100
                face_names.append(name + ' ' + "{:.2f}".format(p))
            else:
                face_names.append("Unknown")

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        print(face_locations.astype(int))
        return face_locations.astype(int), face_names
