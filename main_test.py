import glob
import os
import threading
from threading import Thread

import cv2
import face_recognition
import mediapipe as mp
import time
import numpy as np
from simple_facerec import SimpleFacerec


class FaceDetector():
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

    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []
        cnt = 0
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                print(self.numOfFaces)
                print(id)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                x, y, w, h = bbox
                x1, y1 = x + w, y + h

                if draw:
                    img = self.fancyDraw(img, bbox)

                    # if self.numOfFaces != len(list(self.results.detections)) or len(self.recognized) != self.numOfFaces:
                       # self.recognizeFaces(img)
                    self.numOfFaces = len(list(self.results.detections))
                    name = self.recognizeFaces(img, [(int(x*self.frame_resizing), int(y1*self.frame_resizing), int(x1*self.frame_resizing), int(y*self.frame_resizing))])
                    # else:
                    #     name = self.recognized[cnt]
                    if cnt < len(self.recognized):
                        print(len(self.recognized))
                        cv2.putText(img, f'{name} {int(detection.score[0] * 100)}%',
                                    (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                    2, (255, 0, 255), 2)
                        cnt += 1
        return img, bboxs

    def recognizeFaces(self, img, face_location):
        small_frame = cv2.resize(img, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        # face_locations = face_recognition.face_locations(rgb_small_frame)
        # if self.numOfFaces == len(face_locations):
        #     return self.recognized
        # self.recognized.clear()
        # self.numOfFaces = len(face_locations)

        print(face_location)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_location)
        self.recognized.clear()
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            # matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            self.recognized.clear()
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
        return name + ' ' + "{:.2f}".format(p)
            #     self.recognized.append(name + ' ' + "{:.2f}".format(p))
            # else:
            #     self.recognized.append('Unknown')

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


# define all the kernels size
open_kernel = np.ones((5, 5), np.uint8)
close_kernel = np.ones((7, 7), np.uint8)
dilation_kernel = np.ones((10, 10), np.uint8)


def filter_mask(mask):
    close_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    open_mask = cv2.morphologyEx(close_mask, cv2.MORPH_OPEN, open_kernel)
    dilation = cv2.dilate(open_mask, dilation_kernel, iterations=1)

    return dilation


def main():
    cap = cv2.VideoCapture(0)

    # sfr.save_encoding_images("images/")  #run only once

    print('please show the background now')
    _, background = cap.read()
    time.sleep(4)
    _, background = cap.read()
    pTime = 0
    detector = FaceDetector()
    # sfr.save_encoding_images("images/")  #run only once
    print('Loading images from database...')
    detector.load_data()
    while True:
        success, img = cap.read()


        # cTime = time.time()
        # fps = 1 / (cTime - pTime)
        # pTime = cTime
        # cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # lower bound and upper bound for Green color
        lower_bound = np.array([50, 80, 50])
        upper_bound = np.array([90, 255, 255])

        # find the colors within the boundaries
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Filter mask
        mask = filter_mask(mask)

        # Apply the mask to take only those region from the saved background
        # where our cloak is present in the current frame
        cloak = cv2.bitwise_and(background, background, mask=mask)

        # create inverse mask
        inverse_mask = cv2.bitwise_not(mask)

        # Apply the inverse mask to take those region of the current frame where cloak is not present
        current_background = cv2.bitwise_and(img, img, mask=inverse_mask)

        # Combine cloak region and current_background region to get final frame
        combined = cv2.add(cloak, current_background)

        img, bboxs = detector.findFaces(combined)
        cv2.imshow("Invisible Cloak", combined)
        # cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
