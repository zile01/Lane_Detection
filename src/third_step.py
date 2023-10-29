import numpy as np
import cv2

# Input image dimesions 1280x720 and 960x540
def perspective_transform(binary_image):
    roi_image = binary_image
    roi_image = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)

    input_image_width = binary_image.shape[1]
    input_image_height = binary_image.shape[0]

    output_image_width = input_image_width
    output_image_height = input_image_height

    # Coordinates for source points

    x1, y1 = int(output_image_width/2) - 150, int(input_image_height/2) + 100
    x2, y2 = int(output_image_width/2) + 150, int(input_image_height/2) + 100
    x3, y3 = output_image_width - 50, input_image_height
    x4, y4 = 0 + 50, input_image_height

    # Draw them for debug purposes
    cv2.circle(roi_image, (x1, y1), 5, (0, 0, 255), -1)
    cv2.circle(roi_image, (x2, y2), 5, (0, 0, 255), -1)
    cv2.circle(roi_image, (x3, y3), 5, (0, 0, 255), -1)
    cv2.circle(roi_image, (x4, y4), 5, (0, 0, 255), -1)

    # cv2.imshow("RoI", roi_image)
    # cv2.imwrite('../output_images/perspective_transform_roi.jpg', roi_image)

    # Define the source points (corners of the region you want to transform)
    # These points should be in the order: top-left, top-right, bottom-right, bottom-left
    source_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)

    # Define the destination points for the bird's-eye view
    # These points should be in the order: top-left, top-right, bottom-right, bottom-left
    destination_points = np.array([[0, 0], [output_image_width, 0], [output_image_width, output_image_height], [0, output_image_height]], dtype=np.float32)

    # Calculate the perspective transformation matrix and inverse perspective transformation matrix (used to return back to original point of view)
    perspective_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    perspective_matrix_inverse = cv2.getPerspectiveTransform(destination_points, source_points)

    # Warp the image to create the bird's-eye view
    perspective_transform_img = cv2.warpPerspective(binary_image, perspective_matrix, (output_image_width, output_image_height))

    return perspective_transform_img, perspective_matrix_inverse
