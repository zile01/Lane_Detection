import numpy as np
import cv2

def visualize_lanes(original_img, perspective_transform_img, l_fit, r_fit, Minv):
    new_img = np.copy(original_img)

    if l_fit is None or r_fit is None:
        return original_img

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(perspective_transform_img).astype(np.uint8)
    warp_color = np.dstack((warp_zero, warp_zero, warp_zero))

    h, w = perspective_transform_img.shape
    ploty = np.linspace(0, h-1, num = h)    # to cover same y-range as image
    left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]   #1D array of all x coordinates of points (left lane)
    right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]

    # Transform the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the space between lanes onto the warped blank image
    cv2.fillPoly(warp_color, np.int_([pts]), (0, 69, 255))

    # Draw the lanes onto the warped blank image
    cv2.polylines(warp_color, np.int32([pts_left]), isClosed=False, color=(0, 69, 255), thickness=15)
    cv2.polylines(warp_color, np.int32([pts_right]), isClosed=False, color=(0, 69, 255), thickness=15)

    # cv2.imshow('warp color', warp_color)

    # Return perspective to original one
    original_perspective_img = cv2.warpPerspective(warp_color, Minv, (w, h))
    # cv2.imshow('without add Weighted', original_perspective_img)
    # cv2.imwrite('../output_images/before_overlap_output.jpg', original_perspective_img)

    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, original_perspective_img, 0.5, 0)

    # cv2.imshow('with add Weighted', result)

    return result

def visualize_params(frame, lane_avg_curvature, center_dist):
    text1 = 'Average Lane Curvature: ' + str(lane_avg_curvature) + 'm'
    text2 = 'Distance to Lane Center: ' + str(center_dist) + 'm'

    cv2.putText(frame, text1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 69, 255), 2)
    cv2.putText(frame, text2, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 69, 255), 2)

    return frame