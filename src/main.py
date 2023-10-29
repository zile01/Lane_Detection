# Imports

import first_step
import second_step
import third_step
import fourth_step
import fifth_step
import sixth_step

import numpy as np
import cv2
import sys

# Main - Function that is implementing the pipeline
if __name__ == "__main__":
    do_calibration = False

    # Command line arguments
    n = len(sys.argv) # number of command line arguments

    # There is only 1 flag possible to be passed, so if there is anything, that is going to be potential calibration flag
    if n > 1:
        do_calibration = sys.argv[1]

    # if the passed flag is "-c" -> calibration shall be done
    # otherwise -> we are using the already present calibration params
    if do_calibration == "-c":
        # 1st step - Get Camera Calibration Params and apply them to Undo Distortion of the input frame

        print("Calibration is ongoing...")

        # 1.1 Camera Calibration - Get the calibration matrix & distortion coefficients
        mtx, dist = first_step.camera_calibration()

        # Save them to a file
        np.savez('../camera_cal/calibration_params.npz', mtx=mtx, dist=dist)

        print("Calibration finished")
    else:
        print("Already present calibration parameters used. If there are no params, run program with appropriate flag for calibration: e.x. python3.11 main.py -c")

        # Load the calibration parameters from a file
        calibration_params = np.load('../camera_cal/calibration_params.npz')

        mtx = calibration_params['mtx']
        dist = calibration_params['dist']

    # Input Images
    # frame = cv2.imread(f'../test_images/challange00101.jpg')
    # frame = cv2.imread(f'../test_images/challange00111.jpg')
    # frame = cv2.imread(f'../test_images/challange00136.jpg')
    # frame = cv2.imread(f'../test_images/solidWhiteCurve.jpg')
    # frame = cv2.imread(f'../test_images/solidWhiteRight.jpg')
    # frame = cv2.imread(f'../test_images/solidYellowCurve.jpg')
    # frame = cv2.imread(f'../test_images/solidYellowCurve2.jpg')
    # frame = cv2.imread(f'../test_images/solidYellowLeft.jpg')
    # frame = cv2.imread(f'../test_images/straight_lines1.jpg')
    # frame = cv2.imread(f'../test_images/straight_lines2.jpg')
    # frame = cv2.imread(f'../test_images/test1.jpg')
    # frame = cv2.imread(f'../test_images/test2.jpg')
    # frame = cv2.imread(f'../test_images/test3.jpg')
    # frame = cv2.imread(f'../test_images/test4.jpg')
    # frame = cv2.imread(f'../test_images/test5.jpg')
    # frame = cv2.imread(f'../test_images/test6.jpg')
    # frame = cv2.imread(f'../test_images/whiteCarLaneSitch.jpg')
    # cv2.imshow('Original Image', frame)

    # Path to the input video file
    # cap = cv2.VideoCapture('../test_videos/challenge01.mp4')
    # cap = cv2.VideoCapture('../test_videos/challenge02.mp4')
    # cap = cv2.VideoCapture('../test_videos/challenge03.mp4')
    cap = cv2.VideoCapture('../test_videos/project_video01.mp4')
    # cap = cv2.VideoCapture('../test_videos/project_video02.mp4')
    # cap = cv2.VideoCapture('../test_videos/project_video03.mp4')

    # Stuff for video recording
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output file path and other params: fps, dimensions, etc.
    out = cv2.VideoWriter('output_video.avi', fourcc, fps, (width, height))

    # Processing the each frame of the video
    while True:
    #     brojac += 1
    #     print(brojac)

        ret, frame = cap.read()
        if not ret: break   # Condition to leave the loop

        # cv2.imshow('Raw frame', frame)
        # cv2.imwrite('../output_images/raw_frame_output.jpg', frame)

        # 1.2 Undo Distortion - Apply Calibration parameters in order to get rid of distortion
        undo_distortion_img = first_step.undo_distortion(frame, mtx, dist)
        # cv2.imshow('Image after Undo Distortion', undo_distortion_img)
        # cv2.imwrite('../output_images/undo_distortion_output.jpg', undo_distortion_img)

        # 2nd step - Make the gray scale binary image.

        # 2.1 Binary Image
        binary_img = second_step.binary_image(frame)
        # cv2.imshow('Binary image', binary_img)
        # cv2.imwrite('../output_images/binary_image_output.jpg', binary_img)

        # 3rd step - Transform the view point to bird-eye view.

        # 3.1 Perspective transform
        perspective_transform_img, perspective_matrix_inverse = third_step.perspective_transform(binary_img)
        # cv2.imshow('Perspective Transformed Image', perspective_transform_img)
        # cv2.imwrite('../output_images/perspective_transform_output.jpg', perspective_transform_img)

        # 4th step - Detect lanes and make the approximation of the both left and right lane with a 2nd order polynomial

        # 4.1 Lane Detection
        left_fit, right_fit, return_code, left_lane_inds, right_lane_inds = fourth_step.poly_lane_approx(perspective_transform_img, False)

        # 5th step - Calculate the Params such as: Average Radius of Curvature and Distance of Vehicle to the Lane Center.

        # 5.1 Average Radius of Curvature (in meters)
        lane_avg_curvature = fifth_step.radius_of_curvature(perspective_transform_img, left_lane_inds, right_lane_inds)

        # 5.2 Distance of Vehicle to Lane Center (in meters)
        center_dist = fifth_step.distance_from_center(perspective_transform_img, left_fit, right_fit)

        # 6th step - Transform the view point back to original one and visualize the detected stuff: Lanes and Params

        # 6.1 Visualize Lanes
        lanes_img = sixth_step.visualize_lanes(frame, perspective_transform_img, left_fit, right_fit, perspective_matrix_inverse)
        # cv2.imwrite('../output_images/visualize_lanes_output.jpg', lanes_img)

        # 6.2 Visualize Params
        final_img = sixth_step.visualize_params(lanes_img, lane_avg_curvature, center_dist)
        cv2.imshow('Final Frame', final_img)
        # cv2.imwrite('../output_images/final_frame_output.jpg', final_img)

        out.write(final_img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()