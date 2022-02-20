# import cv2
# import numpy as np
# import time
#
# # replace the red pixels ( or undesired area ) with
# # background pixels to generate the invisibility feature.
#
# ## 1. Hue: This channel encodes color information. Hue can be
# # thought of an angle where 0 degree corresponds to the red color,
# # 120 degrees corresponds to the green color, and 240 degrees
# # corresponds to the blue color.
#
# ## 2. Saturation: This channel encodes the intensity/purity of color.
# # For example, pink is less saturated than red.
#
# ## 3. Value: This channel encodes the brightness of color.
# # Shading and gloss components of an image appear in this
# # channel reading the videocapture video
#
# # in order to check the cv2 version
# print(cv2.__version__)
#
# # taking video.mp4 as input.
# # Make your path according to your needs
# capture_video = cv2.VideoCapture(0)
#
# # give the camera to warm up
# time.sleep(1)
# count = 0
# background = 0
#
# # capturing the background in range of 60
# # you should have video that have some seconds
# # dedicated to background frame so that it
# # could easily save the background image
# for i in range(60):
#     return_val, background = capture_video.read()
#     if return_val == False:
#         continue
#
# background = np.flip(background, axis=1)  # flipping of the frame
#
# # we are reading from video
# while (capture_video.isOpened()):
#     return_val, img = capture_video.read()
#     if not return_val:
#         break
#     count = count + 1
#     img = np.flip(img, axis=1)
#
#     # convert the image - BGR to HSV
#     # as we focused on detection of red color
#
#     # converting BGR to HSV for better
#     # detection or you can convert it to gray
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     # -------------------------------------BLOCK----------------------------#
#     # ranges should be carefully chosen
#     # setting the lower and upper range for mask1
#     lower_red = np.array([202, 0, 0])
#     upper_red = np.array([100, 255, 255])
#     mask1 = cv2.inRange(hsv, lower_red, upper_red)
#     # setting the lower and upper range for mask2
#     lower_red = np.array([170, 120, 70])
#     upper_red = np.array([180, 255, 255])
#     mask2 = cv2.inRange(hsv, lower_red, upper_red)
#     # ----------------------------------------------------------------------#
#
#     # the above block of code could be replaced with
#     # some other code depending upon the color of your cloth
#     mask1 = mask1 + mask2
#
#     # Refining the mask corresponding to the detected red color
#     mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3),
#                                                             np.uint8), iterations=2)
#     mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations=1)
#     mask2 = cv2.bitwise_not(mask1)
#
#     # Generating the final output
#     res1 = cv2.bitwise_and(background, background, mask=mask1)
#     res2 = cv2.bitwise_and(img, img, mask=mask2)
#     final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
#
#     cv2.imshow("INVISIBLE MAN", final_output)
#     k = cv2.waitKey(10)
#     if k == 27:
#         break

# DataFlair Invisible Cloak project using OpenCV.
import threading

import face_recognition
import mediapipe as mp
import cv2
import time
import numpy as np
from simple_facerec import SimpleFacerec

inp = input("Do you want to process live video or a record?\nType 1 for live and 2 for a record: ")

while True:
    if inp == '1':
        cap = cv2.VideoCapture(0)
        break
    elif inp == '2':
        filename = input('Please enter the record name: ')
        cap = cv2.VideoCapture(filename)
        if cap.isOpened() == False:
            raise 'Error opening video file. Please check file path...'
    inp = input('Type 1 for live and 2 for a record: ')

input('Please confirm that you are ready to show the background for a few seconds')

# Store a single frame as background
_, background = cap.read()
time.sleep(2)
_, background = cap.read()

# define all the kernels size
open_kernel = np.ones((5, 5), np.uint8)
close_kernel = np.ones((7, 7), np.uint8)
dilation_kernel = np.ones((10, 10), np.uint8)

sfr = SimpleFacerec()
sfr.load_encoding_images("images/")


# Function for remove noise from mask
def filter_mask(mask):
    close_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    open_mask = cv2.morphologyEx(close_mask, cv2.MORPH_OPEN, open_kernel)
    dilation = cv2.dilate(open_mask, dilation_kernel, iterations=1)

    return dilation

while True:
    ret, frame = cap.read()  # Capture every frame
    # convert to hsv colorspace

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
    current_background = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    # Combine cloak region and current_background region to get final frame
    combined = cv2.add(cloak, current_background)

    cv2.imshow("Invisible Cloak", combined)

    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(combined, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Invisible Cloak", combined)

    if cv2.waitKey(1) == ord('q'):
        break

img = cv2.imread("Messi1.webp")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img2 = cv2.imread("images/Messi.webp")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)

cv2.imshow("Img", img)
cv2.imshow("Img 2", img2)
cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

