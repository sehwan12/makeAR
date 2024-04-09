import numpy as np
import cv2 as cv

# The given video and calibration data
video_file = './data/chessmoving.mp4'
K = np.array([[432.7390364738057, 0, 476.0614994349778],
              [0, 431.2395555913084, 288.7602152621297],
              [0, 0, 1]])
dist_coeff = np.array([-0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 0.0001149909868437517, -0.01803978785585194])
board_pattern = (10, 7)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# Define the 3D points for the sphere object
center = np.array([5 * board_cellsize, 5 * board_cellsize, 0])  # Sphere center at [5, 5] on the chessboard
radius = board_cellsize  # Sphere radius
phi = np.linspace(0, np.pi, 30)
theta = np.linspace(0, 2*np.pi, 30)
phi, theta = np.meshgrid(phi, theta)
x = center[0] + radius * np.sin(phi) * np.cos(theta)
y = center[1] + radius * np.sin(phi) * np.sin(theta)
z = center[2] + radius * np.cos(phi)
sphere_points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break
    width = int(img.shape[1] * 80 / 100)
    height = int(img.shape[0] * 35 / 100)
    dim = (width, height)
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    map1, map2 = None, None
    # Run distortion correction
    if map1 is None or map2 is None:
        map1, map2 = cv.initUndistortRectifyMap(K, dist_coeff, None, None, (img.shape[1], img.shape[0]), cv.CV_32FC1)
        img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw the sphere on the image
        sphere_points_2d, _ = cv.projectPoints(sphere_points, rvec, tvec, K, dist_coeff)
        for point in sphere_points_2d:
            cv.circle(img, tuple(point[0].astype(int)), 1, (0, 255, 0), -1)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec)  # Alternative) `scipy.spatial.transform.Rotation`
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27:  # ESC
        break

video.release()
cv.destroyAllWindows()