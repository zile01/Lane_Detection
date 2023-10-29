import numpy as np
import cv2

# Binary Image
def binary_image(img):
    # Convert original image to HSV format
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV bounds for white
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])

    # HSV bounds for yellow
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # HSV mask for filtering white and yellow
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Extract yellow and white stuff
    masked_image = cv2.bitwise_and(img, img, mask=combined_mask)

    # Convert image to gray scale format
    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # Make binary image: white color will represent white and yellow stuff
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    return binary_image