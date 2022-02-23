import cv2
import time
import numpy as np
from simple_facerec import SimpleFacerec

# inp = input("Do you want to process live video or a record?\nType 1 for live and 2 for a record: ")
#
# while True:
#     if inp == '1':
#         input('Please confirm that you are ready to show the background for a few seconds')
#         cap = cv2.VideoCapture(0)
#         break
#     elif inp == '2':
#         filename = input('Please enter the record name: ')
#         cap = cv2.VideoCapture(filename)
#         if cap.isOpened() == False:
#             raise 'Error opening video file. Please check file path...'
#     inp = input('Type 1 for live and 2 for a record: ')

cap = cv2.VideoCapture(0)
print('please show the background now')
_, background = cap.read()
time.sleep(4)
_, background = cap.read()


# define all the kernels size
open_kernel = np.ones((5, 5), np.uint8)
close_kernel = np.ones((7, 7), np.uint8)
dilation_kernel = np.ones((10, 10), np.uint8)

sfr = SimpleFacerec()
print('Loading images from database...')
#sfr.save_encoding_images("images/")  #run only once
sfr.load_data()

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

cap.release()
cv2.destroyAllWindows()

