import glob
import os
import cv2
import face_recognition
import mediapipe as mp
import time
import numpy as np

class FaceReconizer():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        self.known_face_encodings = []
        self.known_face_names = []
        self.recognized = []
        self.numOfFaces = 0
        self.tolerance = 0.6
        self.frame_resizing = 0.25
        self.face_locations = []
        self.difference = 200

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
        images_path = [fn for fn in glob.glob(os.path.join(images_path, "**/*.*")) if
                       not '.npy' in os.path.basename(fn)]

        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            filepath = img_path.split('.')[0]
            # Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]
            # save the image data to file
            with open(filepath + 'Data.npy', 'wb') as f:
                np.save(f, img_encoding)

        print("Encoding images loaded")

    def getFacesLocations(self, img):
        small_frame = cv2.resize(img, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int)

    def findFaces(self, img):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []

        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)

        face_locations = self.getFacesLocations(img)

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                print(self.numOfFaces)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                x, y, w, h = bbox
                x1, y1 = x + w, y + h

                img = self.fancyDraw(img, bbox)
                bboxs.append([y, x1, y1, x])

            if self.needToRecognizeAgain(self.sort_locations(self.face_locations), self.sort_locations(face_locations)):
                self.recognizeFaces(img)

            for bbox in bboxs:
                name = self.findName(bbox)
                print(name)
                print(self.recognized)
                cv2.putText(img, f'{name} {int(detection.score[0] * 100)}%',
                            (bbox[3], bbox[0] - 10), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)

        return img, bboxs

    def findName(self, bbox):
        print('findName', bbox)
        if len(self.recognized) == 0 or len(self.face_locations) == 0:
            return ""
        distances = []
        for v in self.face_locations:
            distances.append(np.linalg.norm(v - bbox))

        print('findName', min(distances))
        if min(distances) > self.difference:
            return 'Unknown'
        return self.recognized[np.argmin(distances)]

    def sort_locations(self, locations):
        """

        :param locations: the vectors to sort
        :return: list of vectors sorted by size of vector
        """
        sizes = []
        for loc in locations:  # calculate the size of each vector
            sum = np.square(loc[0]) + np.square(loc[1]) + np.square(loc[2]) + np.square(loc[3])
            size = np.sqrt(sum)
            sizes.append((loc, size))

        lst = len(sizes)
        for i in range(0, lst):  # bubble sort by size
            for j in range(0, lst - i - 1):
                if (sizes[j][1] > sizes[j + 1][1]):
                    temp = sizes[j]
                    sizes[j] = sizes[j + 1]
                    sizes[j + 1] = temp
        return [tup[0] for tup in sizes]

    def needToRecognizeAgain(self, V, B):
        """

        :param V: list of vectors
        :param B: list of vectors
        :return: True if the distance between vi to bi is greater than self.difference
        """
        if len(V) != len(B):
            return True
        print(V)
        print(B)

        if len(V) == 0:
            return True
        dist = -1
        for v, b in zip(V, B):
            dist = np.linalg.norm(v - b)
            if dist > self.difference:
                print('needToRecognizeAgain', dist)
                return True
        print('needToRecognizeAgain', dist)
        return False

    def recognizeFaces(self, img):
        small_frame = cv2.resize(img, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        self.recognized.clear()

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            # matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            # get the distances between faces from data to face from the video
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)  # best match found
            if face_distances[best_match_index] <= self.tolerance:
                name = self.known_face_names[best_match_index]  # get the name of this match
                p = (1 - face_distances[best_match_index]) * 100
            else:
                name = 'Unknown'
                p = 0
            self.recognized.append(name)
            print(f'{name} {p * 100}%')
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        self.face_locations = face_locations.astype(int)
        print('recognizeFaces', self.face_locations)

    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, (x1, y1), (x, y), (0, 0, 200), 4)
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
        # Top Right  x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # Bottom Left  x,y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img

