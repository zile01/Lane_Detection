import numpy as np

# Detected Params are the params detected by camera sensor during the driving
# In this case there are: Average Radius of Curvature and Distance of Vehicle to Lane Center

# Curvature Radius and Distance from Center should be measured in meters, so transformation from pixels to meters shall be done
ym_per_pix = 30.0/720.0
xm_per_pix = 3.7/700.0

def radius_of_curvature(bin_img, l_lane_inds, r_lane_inds):

    # Define y-value where we want radius of curvature (bottom of the image)
    y = bin_img.shape[0] - 1

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Get the points (x, y) of the pixels around the left and right line on the same way as for curve fitting
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds]

    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]

    if len(leftx) != 0 and len(rightx) != 0:
        # Fit the new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        # Calculate the new radius of a curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])

        # Calculate the average curvature
        lane_avg_curvature = (left_curverad + right_curverad)/2.0
    else:
        lane_avg_curvature = -1

    return lane_avg_curvature

def distance_from_center(image, left_fit, right_fit):
    # Define y-value where we want radius of curvature (bottom of the image)
    y = image.shape[0]

    # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts
    if right_fit is not None and left_fit is not None:
        car_position = image.shape[1]/2
        l_fit_x_int = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
        r_fit_x_int = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center_dist = np.absolute((car_position - lane_center_position) * xm_per_pix)
    else:
        center_dist = -1

    return center_dist