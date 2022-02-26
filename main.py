import numpy as np
import cv2
import time
from FaceRecognizer import FaceReconizer

# define all the kernels size
open_kernel = np.ones((5, 5), np.uint8)
close_kernel = np.ones((7, 7), np.uint8)
dilation_kernel = np.ones((10, 10), np.uint8)


def filter_mask(mask):
    close_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    open_mask = cv2.morphologyEx(close_mask, cv2.MORPH_OPEN, open_kernel)
    dilation = cv2.dilate(open_mask, dilation_kernel, iterations=1)

    return dilation

def userInteraction():
    answer = input('Do you want to decipher an image? if you want to decipher live video choose No Y/N: ')
    if answer.lower() == 'y':
        answer = input('Enter the path of the image: ')
        return answer
    return 'n'

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1100)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1100)

    recognizer = FaceReconizer()
    recognizer.save_encoding_images("images/")  #run only once if there are new images
    #print('Loading images from database...')
    recognizer.load_data()
    path = userInteraction()
    if path != 'n':
        try:
            recognizer.imageRecognizer(path)
            print('Type q to quit...close the window for continue to live cam')
            while cv2.waitKey(1) != ord('q'):
                continue
            return
        except:
            print('Image does not exist or path is wrong')

    answer = input('Do you want the invisible cloak effect? Y/N: ')
    invisible_cloack = False
    if answer.lower() == 'y':
        input('Please confirm that you are ready to show the background now, once you confirmed you have 4 seconds: ')
        for i in range(1, 5):
            _, background = cap.read()
            time.sleep(1)
            print(i, '...')
        _, background = cap.read()
        invisible_cloack = True
    while True:
        success, img = cap.read()
        combined = img
        if invisible_cloack:
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

        img = recognizer.findFaces(combined)
        cv2.imshow("Live cam", combined)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
