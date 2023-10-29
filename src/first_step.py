import numpy as np
import cv2

# Camera Calibration
def camera_calibration():
    # Define the chessboard rows and columns - Empiricaly determined
    # e.g. 6:9 ratio works better for matching, but the end results are much worse
    rows = 5
    cols = 9

    # Set the termination criteria for the corner sub-pixel algorithm
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
    objectPoints = np.zeros((rows * cols, 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    # Create the arrays to store the object points and the image points
    objectPointsArray = []
    imgPointsArray = []

    # Loop over the image files
    for i in range(1, 21):
        # Load the image
        img = cv2.imread(f'../camera_cal/calibration{i}.jpg')

        # Convert it to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners - TODO very slow function, can be speeded up by rescaling input images
        ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

        # Make sure the chessboard pattern was found in the image
        if ret:
            # Refine the corner position
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Add the object points and the image points to the arrays
            objectPointsArray.append(objectPoints)
            imgPointsArray.append(corners)

            # Draw the corners on the image
            # cv2.drawChessboardCorners(img, (rows, cols), corners, ret)

            # Display the image
            # cv2.imshow('chessboard', img)

    # Calibrate the camera and save the results
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)

    return mtx, dist

# Undo Distortion
def undo_distortion(img, mtx, dist):

    # Dimensions of the input image
    h, w = img.shape[:2]

    # Obtain the new camera matrix and undistort the image
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

    return undistorted_img
