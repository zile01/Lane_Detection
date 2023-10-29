# Subject: Multimedijalni sistemi u automobilskoj industriji

# Project: Lane Finding Project

# Year: 2023/24

# Author: Pavle Vuković, E2 76/2023

## Project Text:
The goals / steps of this project are the following:
 * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
 * Apply a distortion correction to raw images.
 * Use color transforms, gradients, etc., to create a thresholded binary image.
 * Apply a perspective transform to rectify binary image ("birds-eye view").
 * Detect lane pixels and fit to find the lane boundary.
 * Determine the curvature of the lane and vehicle position with respect to center.
 * Warp the detected lane boundaries back onto the original image.
 * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[RawFrame]: ./output_images/raw_frame_output.jpg "Raw Frame"
[UndoDistortionFrame]: ./output_images/undo_distortion_output.jpg "Undo Distortion Frame"
[BinaryImage]: ./output_images/binary_image_output.jpg "Binary Image"
[PerspectiveTransformRoI]: ./output_images/perspective_transform_roi.jpg "Perspective Transform RoI"
[PerspectiveTransformFrame]: ./output_images/perspective_transform_output.jpg "Perspective Transform Output"
[LaneDetectionFrame]: ./output_images/lane_detection_output.png "Lane Detection Frame"
[BeforeOverlapFrame]: ./output_images/before_overlap_output.jpg "Before Overlap Frame"
[VisualizeLanesFrame]: ./output_images/visualize_lanes_output.jpg "Visualize Lanes Frame"
[FinalFrame]: ./output_images/final_frame_output.jpg "Final Frame"

## Config
Within */project* repository there are several sub-repositories:
 * *camera_cal* - repository contains stuff about camera calibration:
	* camera calibration input images (e.g. "*project/camera_cal/calibration1.jpg*")
	* camera calibration parameters: calibration matrix and distortion coefficients (e.g. "*project/camera_cal/calibration_params.npz*")
* *output_images* - repository contains images that are output of appropriate solution steps.
* *output_videos* - repository contains videos that are output of appropriate solution steps.
 * *reference_images* - repository contains images used as a reference in this document.
 * *src* - repository contains source code of the project:
	 * main.py
	 * first_step.py
	 * second_step.py
	 * third_step.py
	 * fourth_step.py
	 * fifth_step.py
	 * sixth_step.py
 * *test_images* - repository contains images used as an input for testing.
 * *test_videos* - repository contains videos used as an input for testing.
## Running
In order to run a project, a few step should be done.
 * Open terminal and position yourself in repository: *project/src/*.
 * Run following command: *python{**version**} main.py {**flags**}*
	* ***{version}*** should be replaced with a current python version (e.g. python3.11)
	* ***{flags}*** at the moment only: "-c" flag is being used. It stands for calibration.
	If you want to perform camera calibration you shall include this flag. (e.g. python3.11 main.py -c)
	Otherwise, if you want to use pre-calculated camera calibration parameters, exlude this flag. (e.g. python3.11 main.py)

**NOTE:** Camera Calibration could take a while (up to 30 seconds) so it is recommended to include "-c" flag only at first application executing.

**NOTE:** In order to change input video, enter the *project/src/main.py* file and uncomment the appropriate line: e.g. `cap = cv2.VideoCapture('../test_videos/project_video01.mp4')`. All other lines should be commented out.

**NOTE:** Solution tested on python version: 3.11

## Implementation
Implementation is divided into few steps:
 * 1st step - Get Camera Calibration Params and apply them to Undo Distortion of the input frame
 * 2nd step - Make the gray scale binary image
 * 3rd step - Transform the view point to bird-eye view
 * 4th step - Detect lanes and make the approximation of the both left and right lane with a 2nd order polynomial
 * 5th step - Calculate the Params such as: Average Radius of Curvature and Distance of the Vehicle to the Lane Center
 * 6th step - Transform the view point back to original one and visualize the detected stuff: Lanes and Params

### 1st Step - Get Camera Calibration Params and apply them to Undo Distortion of the input frame
file: *project/src/first_step.py*

There are 2 major parts of this step:
 * Camera Calibration
 * Undo Distortion

#### Camera Calibration
function: *first_step.camera_calibration*

return value(s):
* `mtx` - calibration matrix
* `dist` - distortion coefficients

**NOTE:** Other params such as: rotation matrix and translation matrix can be calculated too. However, they are not used.

Steps:
* prepare the object points - Object points are 3D world points with known positions.
* loop - iterate through input calibration images (chessboard images). Image is loaded and converted to gray scale.
* find corners - corners are found in input images using the: `cv2.findChessboardCorners` function. Its position is slightly corrected using the: `cv2.cornerSubPix` function. Corners are image points. They are the image representation of 3D world points (Object Points).
* calibrate camera - It is done by using the `cv2.calibrateCamera` function. It uses the image and object points and based on them finds the calibration params: calibration matrix, distortion coefficients, rotational matrix and translation matrix.

**Important Parameters:** `rows` and `cols` suggest which chessboard pattern to look for when finding the corners in input images. (e.g. chessboard pattern with 5 rows and 9 columns).

Current values for important parameters:
|rows|cols|
|----|----|
| 5  | 9  |

**Possible Improvement:**
By variating the `rows` and `cols` parameters, there is different successful rate when finding the chessboard pattern, thus corners on imput images.

E.g. when using the ratio of 6:9 (rows : cols), pattern is found almost on every input image.

However calibration params calculated after and used for undo distortion are not so good as calibration params calculated on 5:9 ratio (which has less succesfull rate on pattern finding). The space for improvement is finding the ratio rows : cols where calibration params are going to give the best results for undo distortion.

**Output**
Output is the file: *project/camera_cal/calibration_params.npz*. It contains calibration params: calibration matrix and distortion coefficients.

**References:**
 * https://github.com/chedadsp/dosivuav-py

#### Undo Distortion - This is the process of applying the calibration parameters in order to undo the distortion effect on input frames.
function: *first_step.undo_distortion*

return value(s):
* `undo_distortion_img`- undistorted image.

Steps:
* form new camera matrix - It is done i function: `cv2.getOptimalNewCameraMatrix`.
* undo distrotion - It is done in function: `cv2.undistort`. It uses the camera calibration params and new camera matrix to undistort the input frame.

**Output:**

![alt text][RawFrame]
![alt text][UndoDistortionFrame]

**References:**
* https://github.com/chedadsp/dosivuav-py

### 2nd Step - Make the gray scale binary image

file: *project/src/second_step.py*

#### Binary Image
function: `second_step.py.binary_image`

return value(s):
* `binary_img` - gray scaled binary image.

Idea is to make gray scale binary image, having only two intensities on picture: 255 for features on input image that are white or yellow and 0 for everything else.

Steps:

* color representation convertion - HSV color representation is being used. Input image is converted from BGR to HSV color representation using the: `cv2.cvtColor`.
* lower and upper bounds creation for both colors. Bounds - lower and upper are created for both yellow and white colors (e.g. `lower_white  =  np.array([0, 0, 200])` represents lower bound for white color). Every bound has the H (hue), S (saturation) and V (value) component.
* mask creation - By combining two bounds in function `cv2.inRange`, masks are created. By combining different masks it is possible to create only one mask. In this case this is the mask created by combining the yellow and white mask using `cv2.bitwise_or` .
* features extracting - When the singular mask is created it is possible to extract features from input image using it. It is done by: `cv2.bitwise_and`. Now everything that is not yellow or white is colored in black.
* gray scale image creation - Image is converted from HSV (3 canals) to gray scale (1 canal) using the: `cv2.cvtColor`.
* binary image creation - It is done by using the: `cv2.threshold` function. Extracted yellow and white stuff are now all white (255) and everything else is black (0).

**Important Parameters**
The params of interest are the ones forming the bounds (lower and upper). Current params are:

|bound       |Hue|Saturation|Value|
|------------|---|----------|-----|
|white-lower |0  |0         |200  |
|white-upper |255|30        |255  |
|yellow-lower|20 |100       |100  |
|yellow-upper|30 |255       |255  |

**Possible improvements**
Current params do the job of extracting yellow and white stuff good enough. In some cases of greater brightness or darkness, they can get us in situation of extracting a lot of noise. By finding better param values the extraction process can be improved. This problem is not only solvable by finding better param values but also by using the other advanced techiques: noise reduction, filtering based on brightness, etc.

**Output**

![alt text][BinaryImage]

**References**
* https://github.com/chedadsp/dosivuav-py

### 3rd Step -  Transform the view point to bird-eye view
file: *project/src/third_step.py*
#### Perspective Transform
function: `third_step.perspective_transform`

return value(s):
* `perspective_transform_img` - transformed view point of input frame to bird-eye view point.
* `perspective_matrix_inverse` - inverse matrix of view point transformation. It is going to be used in inverse perspective transform.

Idea is to get the road view from bird's perspective.

Steps:

* setting the source points - These are the points of the input picture (gray scale binary image) that has to be mapped.
* setting the destination points - These are the points of the output image. Source points are mapped to destination points. This mapping is the process of perspective transforming.
* calculating the perspective transform matrix - Perspective transform matrix is calculated both for perspective transform and for inverse perspective transform. This is done with function: `cv2.getPerspectiveTransform`. The difference for "normal" and inverse matrix is the order of the arguments. It is inverse, thus creating the inverse matrix.
* perspective transform - Transforming the original view point to bird eye view point is done with function: `cv2.warpPerspective`.

**Important Parameters**
Params of interest are the source and destination points. Source points represent a region of interest. This is the region on the input image that is valuable and which is going to be transformed.

![alt text][PerspectiveTransformRoI]

The red points on the image represent the source points. These are hardcoded based on the input image dimensions:

|source points|x                                |y                                |
|-------------|---------------------------------|---------------------------------|
|top left     |`int(output_image_width/2) - 150`|`int(input_image_height/2) + 100`|
|top right    |`int(output_image_width/2) + 150`|`int(input_image_height/2) + 100`|
|bottom right |`output_image_width  -  50`      |`input_image_height`             |
|bottom left  |`0 + 50`                         |`input_image_height`             |

On the other side there are destination points which coordinates are:

|destination points|x                   |y                    |
|------------------|--------------------|---------------------|
|top left          |`0`                 |`0`                  |
|top right         |`output_image_width`|`0`                  |
|bottom right      |`output_image_width`|`output_image_height`|
|bottom left       |`0`                 |`output_image_height`|

**Possible Improvements**
Source points are hardcoded based on the image dimensions. This is flawed approach and work only in special circumstances. Potential problems are: different camera resolution, different camera positions on vehicle, etc. There should be much more clever algorithm for generating the source points.

**Output**

![alt text][PerspectiveTransformFrame]

**References**
* https://chat.openai.com
* https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html

### 4th step - Detect lanes and make the approximation of the both left and right lane with a 2nd order polynomial

file: *project/src/fourth_step.py*

#### Lane Detection
function: `fourth_step.poly_lane_approx`

return value(s):
* `left_fit`- coefficients of polynomial which is approximating the left lane.
* `right_fit`- coefficients of polynomial which is approximating the right lane.
* `return_code`- code which is telling us if the function has been succesfully executed (return_code = 0) or there is no detection of left lane (return_code = -1) or right lane (return_code = 1).
* `left_lane_inds`- the list of indeces that are representing the pixels (x and y coordinate) that are near the left lane.
* `right_lane_inds`- the list of indeces that are representing the pixels (x and y coordinate) that are near the right lane.

The function is finding the polynomial (2nd order) coefficients (a, b and c) for both left and right lane. It also finds the pixels that are near lanes.

Steps:

* histogram calculation - this is not a conventional histogram. It is calculated in order to find the peaks of intensity both on left and right side of the image. It matters for first iteration of algorithm. Peaks of histogram represent the starting point of both windows.
* setting the window parameters - In each frame there are n windows on both sides of image. Window params are: height, margin (height/2), number of windows.
* identification of all non zero pixels on image - Finding the x and y axis coordinates of all non zero (non black) pixels on image. This matters because the subset of these pixels will be the nonzero pixels around each lane.
* loop execution - Loop is executing until all windows are applied to each frame. By default there are 9 windows for each frame, thus 9 loop iterations. Wihin loop there are steps:
	* find windows bounds - determine the:
		* `win_xleft_low` - minimal x value for left window.
		* `win_xleft_high` - maximal x value for left window.
		* `win_xright_low` - minimal x value for right window.
		* `win_xright_high` - maximal x value for right window.
		* `win_y_low` - minimal y value.
		* `win_y_high` - maximal y value.

		These are the params that determine the window position in each iteration. E.g. win_y_low in first iteration will be the: y dimension of the image minus height of the window and win_y_high will be the height of the window. In next iteration both bounds will be decreased by the size of the window until all iterations are done.
	* find indeces of pixels that are within a window - idea is to find the indeces of elements from `nonzerox` and `nonzeroy` arrays that are within windows. Using the x coordinate from `nonzerox` and y from `nonzeroy` the nonzero pixel within window is formed, so it is possible to find all non zero pixels within each window.
	* find new starting point for next iteration - in first iteration the starting points of the windows were determined by histogram. Now it is the mean of x values of pixels found within each window.

* find all the point close to lanes - based on indeces found in the loop and a set of all nonzero points in the frame, it is possible to find the subset of all nonzero points in the proximity of the lane.
* find the 2nd order polynomial for both lanes - Using the function: `cv2.polyfit`, it is possible to find the coefficients of the nth order polynomial which passes through all the points (x and y coordinates) given as input (points within windows, thus near lanes, calculated in previous steps).

**Important parameters**

    nwindows = 9
    margin = 100
    minpix = 50

* nwindows - Determines how many windows will be used in each frame.
* margin - Determines the width of each windows. Margin is the number which determines how many pixels both left and right around the window starting point there is.
* minpix - Determines how many pixels within windows is a minimum in order to calculate new starting point on x axis. If there are not enough pixels within window, the starting point remains the same.

**Output**
Picture shows the output of algorithm after one frame is processed. Blue polygons are windows. Yellow lines are polynomials that represent the lanes.

![alt text][LaneDetectionFrame]

**References**
* https://chat.openai.com
* https://github.com/arturops/self-drivingCar/tree/master/AdvancedLaneFinding

### 5th step - Calculate the Params such as: Average Radius of Curvature and Distance of the Vehicle to the Lane Center

file: *project/src/fifth_step.py*

#### Average Radius of Curvature
function: `fifth_step.radius_of_curvature`

return value(s):
* `lane_avg_curvature` - average radius of curvature (in meters).

The radius of curvature is calculated by formula:

    R = ((1 + f'(y))^2)^3/2 / f''(y)

In this case, f(y) is the 2nd order polynomial approximation of lane(s). So there will be 2 radiuses of curvature, one for each lane.

Values of the first and second derivative can be measured using the 2nd order polynomial coefficients.

    f(y)   = A*y^2 + B*y + C
    f'(y)  = 2*A*y + B
    f''(y) = 2*A

Average radius of curvature is calculated by finding the average value between radius of curvature for left and right lane.

    Ravg = (Rleft + Rright) / 2

NOTE: Radius of curvature should be measured in meters, so it is mandatory to convert pixel values to meter values. This is done by using the scalars: `xm_per_pix` and `ym_per_pix`.

Steps:
* find the 2nd order polynomial coefficients - Using the indeces both for left and right lane (indices came from previous step) it is possible to select subset of `nonzerox` and `nonzeroy`, thus getting the pixels that belongs to lanes. Then those pixels are multiplied by appropriate scalar in order to convert them to meters. Using the `np.polyfit` on the same way as in the previous step, 2nd order polynomials for approximating the lanes are calculated. Coefficients would not have to be calculated if there is no convertion from pixels to meters.
* calculate the radius of curvature for both left and right lanes - By applying the formula from above, radius of curvature is calculated.
* calculate the average radius of curvature: By applying the formula from above, average radius of curvature is calculated.

**Important parameters**
|ym_per_pix|xm_per_pix|
|----------|----------|
|30.0/720.0|3.7/700.0 |

`ym_per_pix` - 30 meters in real life represent 720 pixels on the y axis on the image.

`xm_per_pix` - 3.7 meters (average lane width) in real life represent 700 pixels on the x axis of the image.

**Output**
Output of this step is the value of the average radius of curvature. It is being used for next step.

**References**
* https://github.com/arturops/self-drivingCar/tree/master/AdvancedLaneFinding

#### Distance of the Vehicle to the Lane Center
function: `fifth_step.distance_from_center`

return values(s):
* `center_dist` - distance of vehicle to the lane center (in meters).

The Distance of the Vehicle to the Lane Center is calculated by formula:

    d = car position - lane center position

* car position represents the middle of the input frame. The assumption is: the camera is placed in the middle of the car horizontal axis.
* lane center position represents the middle of the distance between two lanes.

Steps:
* calculate the car position - Divide the width of the frame by 2.
* calculate the lane center position - This is done by following formula:

	    lane center position = (xleft + xright) / 2

	xleft and xright are calculated using the 2nd order polynomial approximation for both left and right lanes. X values are calculated when appropriate y value is put as an input to the polynomial functions. In this case the y value used is the max y value or the height of the frame.
* calculate the distance of the vehicle to the lane center - Using the formula from above and values from previous steps, the distance is calculated. In this case too, the convertion from pixels to meters is done by applying the appropriate parameter.

**Output**
Output of this step is the value of the distance of the vehicle to the lane center. It is going to be used in next step.

**References**
* https://github.com/arturops/self-drivingCar/tree/master/AdvancedLaneFinding

### 6th step - Transform the view point back to original one and visualize the detected stuff: Lanes and Params
file: `project/src/sixth_step.py`
#### Visualize Lanes
function: `sixth_step.visualize_lane`

return values(s):
* `lanes_img` - original frame with detected lanes visualized.

Steps:
* prepare points(`pts`) in appropriate format - `pts` are basically all the points (pixels) with x and y coordinate that belong to lanes. These are calculated using the polynomial functions - all y coordinates of the image are used as an input for both polynomial functions, thus generating all the x coordinates.
* draw the space between lanes on the image - The space between two lanes is drawn using the: `cv2.fillPoly` function. The space between lanes is drawn to the blank image in warped perspective.
* draw the  lanes on the image - The lanes are drawn using the: `cv2.polylines` function. The  lanes are drawn to the same image as in previous step and still in warped perspective.
* return perspective to the original one - The perspective of the blank image with lanes and space between them is transformed back to the original one using the: `cv2.warpPerspective` function. Important thing is that the matrix of transform is the inverse one calculated in 3rd step.

![alt text][BeforeOverlapFrame]

* combine the original frame and transformed image from previous step - Using the: `cv2.addWeighted` it is possible to overlap the original frame with the image from the previous step. Now, the lanes and space between them are drawn on the original frame.

**Output**

![alt text][VisualizeLanesFrame]

**References**
* https://github.com/arturops/self-drivingCar/tree/master/AdvancedLaneFinding

#### Visualize Params
function: `sixth_step.visualize_params`

return value(s):
* `final_img` - the final image with visualized lanes and params.

Steps:
* visualize the previously calculated params: average radius of the curvature and the distance of the vehicle to the lane center - this is done by using the: `cv2.putText` function with appropriate params.

**Output**
The final image looks like:

![alt text][FinalFrame]

**References**
* https://chat.openai.com

## Final Video Output
After all the previous steps are applied to the input video (in this case *project/test_videos/project_video01.mp4*), results are as following:

<video width="960" height="540" controls>
  <source src="./output_videos/output_video.mp4" type="video/mp4">
  bla
</video>

## Conclusion

The goal of detecting lanes and finding the average curvature and distance to the lane center has been successfully achieved. The solution is tested using the input images and videos. Results are good enough but not ideal.

**Problems**

The biggest problems are seen when conditions vary from being "normal". Few cases:

* turning is too harsh - This case is pretty visible in video: *project/test_videos/challenge03.mp4*. Problems are there because default RoI (hardcoded), used in perspective transform force us to process to much of a space in y direction, thus processing the space which is out of the lanes. The RoI should be adaptively selected in order to find the part of the road of interest.
* too much of a noise - In situation where, on the road or in the close proximity of it, there is a lot of white or yellow noise (e.g. yellow signs or yellow and white cars) the histogram and lane detection algorithm does not work appropriately. The fix could be to filter the noise when needed. The tricky part is to know when the filter appliance is needed.
* brightness - In situations where brightness is too high or too low there are problems resulting in too much of noise on the road. The solution here could be to darken the image if the brightness level is too high and to increase brightnes in other scenario. Brightness level could be measured using the histogram.
* etc. These are the most obvious problems but certainly there are the others.

**Future Work**

Test phase has not been thorough enough. It is important to test the current algorithm in other test situations. Current problems are there, but it is important to find other, currently not visible problems. Then, algorithm for lane detection should be modified in order to fix above mentioned problems.
