from operator import le
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import glob

# define width and height for training images
IMAGE_WIDTH = 70
IMAGE_HEIGHT = 70

SHOW_STEPS = False

# define our characters that we want to train
valid_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'D', 'Gh', 'H', 'J', 'L', 'M', 'N', 'P',
               'PuV', 'PwD', 'Sad', 'Sin', 'T', 'Taxi', 'V', 'Y']


# main train function
def main():
    # create a numpy array with zero member and size of train images
    npa_flattened_images = np.empty((0, IMAGE_WIDTH * IMAGE_HEIGHT))
    # array that holds our classifications
    int_classifications = []

    # generate images for each character
    for char in valid_chars:
        images = [cv2.imread(file) for file in glob.glob(f'iranian_plate_dataset/{char}/*.jpg')]
        counter = 0

        for image in images:
            counter += 1

            print('CHAR ' + char + " -> " + str(counter) + '/' + str(len(images)))

            # convert to grayscale for easier processing
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # create black & white image
            image_thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                                 11, 2)
            # find image contours to do the process and train on that data
            npa_contours, npa_hierarchy = cv2.findContours(image_thresh.copy(), cv2.RETR_EXTERNAL,
                                                           cv2.CHAIN_APPROX_SIMPLE)

            # declare some variables to hold bindings of each character
            # Note: in Farsi some character contours are not connected to each other so we need to do this step
            min_x, min_y, max_x, max_y = IMAGE_WIDTH, IMAGE_HEIGHT, 0, 0

            # Uncomment these lines if you want to see character bindings
            # cv2.drawContours(image, npa_contours, -1, (0, 255, 0), 1)
            # cv2.imshow('test2', image)

            # find contours of possible characters
            for c in npa_contours:
                [x, y, w, h] = cv2.boundingRect(c)
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x + w > max_x:
                    max_x = x + w
                if y + h > max_y:
                    max_y = y + h

            # use each character ord code for classifications
            int_char = ord(char.replace('.png', ''))
            int_classifications.append(int_char)

            # crop character part of the image and stretch it to original size
            image_thresh = image_thresh[min_y:max_y, min_x:max_x]
            image_char_resized = cv2.resize(image_thresh, (IMAGE_WIDTH, IMAGE_HEIGHT))

            # reshape the image intro numpy array, so we could save it into the training file
            npa_flattened_image = image_char_resized.reshape((1, IMAGE_WIDTH * IMAGE_HEIGHT))
            # append reshaped image intor trained array
            npa_flattened_images = np.append(npa_flattened_images, npa_flattened_image, 0)

            if SHOW_STEPS:
                cv2.imshow('test', image_char_resized)
                cv2.waitKey(0)

    # declare a float shaped numpy array and append our classifications to it
    float_classifications = np.array(int_classifications, np.float32)
    npa_classifications = float_classifications.reshape((float_classifications.size, 1))

    print("Training complete !!")

    # write classifications to file
    np.savetxt("classifications/classifications.txt", npa_classifications)
    # write flattened images to file
    np.savetxt("classifications/flattened_images.txt", npa_flattened_images)

    cv2.destroyAllWindows()  # remove windows from memory

    return


if __name__ == "__main__":
    main()
