'''
Webcam Augmented Reality
@author: Samuel Veloso - samvrocks
'''
import numpy as np
import cv2

# Termination criteria
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

'''
Returns the extrinsic, intrinsic, and distortion parameters from your web-cam.

Input parameters:
    n_frames : Number of frames needed to calibrate the camera (10 at least)
    
Ouput parameters:
    rms : The root mean square re-projection error
    cam_matrix : 3x3 matrix with focal length and optical centers parameters
    dist_coefs : Distortion coefficients given by [k1 k2 p1 p2 p3]
    r_vecs : Rotation vectors
    t_vecs : Translation vectors
'''
def calibrateWebcam(w_board, h_board, n_frames):
        
    # Arrays to store object points and image points from all the images
    P_obj = [] # 3D point in real world space
    P_img = [] # 2D points in image plane
    
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    object_points = np.zeros((w_board*h_board, 3), np.float32)
    object_points[:,:2] = np.mgrid[0:w_board, 0:h_board].T.reshape(-1, 2)
    
    # Enable web-cam to capture images
    cam_handler = cv2.VideoCapture(0)
    
    print str(n_frames) + ' frames are needed!'
    print 'Press key \'c\' to capture a frame...'
    
    # Capture N patterns
    while n_frames > 0 :
        
        # Capture a frame
        _, cam_frame = cam_handler.read()
            
        # Get 8bit frame
        gray_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        chessboard_found, corners = cv2.findChessboardCorners(gray_frame, (w_board, h_board), None)
        
        # If found, add object points, image points (after refining them)
        if chessboard_found == True :
            # Refine the corners
            cv2.cornerSubPix(gray_frame, corners, (11, 11), (-1, -1), CRITERIA)
            
            # Overlay corners points
            cv2.drawChessboardCorners(cam_frame, (h_board, w_board), corners, chessboard_found)
            
            # Capture a frame if press key 'c'
            if cv2.waitKey(1) & 0xFF == ord('c') :
                # Store object and image points correspondence
                P_obj.append(object_points)
                P_img.append(corners)
                # Decrement the number of remaining frames
                n_frames -= 1
                print 'Frame captured! Now only ' +  str(n_frames) + ' more frames are needed...'
            
        cv2.imshow("Webcam calibration", cv2.flip(cam_frame, 1)) 
        if cv2.waitKey(1) & 0xFF == ord('q') : quit()
        
    cam_handler.release()
    cv2.destroyAllWindows()
            
    return cv2.calibrateCamera(P_obj, P_img, gray_frame.shape[::-1], None, None)

'''
Returns the undistorted image applying camera matrix and distortion parameters

Input parameters:
    distorted_img : Distorted image to undistort
    cam_matrix : 3x3 matrix with focal length and optical centers parameters
    dist_coefs : Distortion coefficients given by [k1 k2 p1 p2 p3]
    alpha : Scaling parameter
    
Ouput parameters:
    distorted_img : The undistorted image
'''
def undistortImage(distorted_img, cam_matrix, dist_coefs, alpha):

    # Get distorted image dimensions
    height, width = distorted_img.shape[:2]
    
    # Refine the camera matrix using alpha as free scaling parameter
    new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coefs, (width, height), alpha, (width, height))
    
    # Get undistorted image
    undistorted_img = cv2.undistort(distorted_img, cam_matrix, dist_coefs, None, new_cam_matrix)
    
    # Crop the undistorted image generated
    x_ini, y_ini, width, height = roi
    undistorted_img = undistorted_img[y_ini:y_ini + height, x_ini:x_ini + width]
    
    return undistorted_img

'''
Compute the 3x3 homography matrix using as correspondence:
    - 4 corners of source image        ->    (tl_i, tr_i, bl_i, br_i)
    - 4 corners of the checkerboard    ->    (tl_c, tr_c, bl_c, br_c)

Input parameters:
    src_image : Source image
    board_corners : All corners of the checkerboard
    w_board : Checkerboard width (in tiles)
    h_board : Checkerboard height (in tiles)
    
Ouput parameters:
    distorted_img : The undistorted image
'''
def getHomographyMatrix(src_image, board_corners, w_board, h_board):
      
    # Resize input corners
    board_corners = board_corners[:,0,:]
        
    # Board <-> Image correspondece
    board_rect = getCheckerboardRect(board_corners, w_board, h_board)
    image_rect = getImageRect(src_image)
    
    # Get 3x3 homography matrix
    h_matrix = cv2.findHomography(image_rect, board_rect)[0]
    
    return h_matrix

'''
Perform planar based augmented reality (AR), overlay an image on top
of the webcam which is aligned with the checkerboard

Input parameters:
    img_to_augment : Image to overlay on top of the checkerboard
    cam_matrix : 3x3 matrix with focal length and optical centers parameters
    dist_coefs : Distortion coefficients given by [k1 k2 p1 p2 p3]
    w_board : Checkerboard width (in tiles)
    h_board : Checkerboard height (in tiles)
    alpha : Scaling parameter
'''
def playAugmentedReality(img_to_augment, cam_matrix, dist_coefs, w_board, h_board, alpha=1):    
    
    # Enable web-cam to capture images
    cam_handler = cv2.VideoCapture(0)

    # Capture N patterns
    while True :
        
        # Capture a frame
        cam_frame = cv2.flip(cam_handler.read()[1], 1)
        
        # Undistort frame
        undistorted_cam_frame = undistortImage(cam_frame, cam_matrix, dist_coefs, alpha)
            
        # Get 8bit frame
        gray_frame = cv2.cvtColor(undistorted_cam_frame, cv2.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        chessboard_found, corners = cv2.findChessboardCorners(gray_frame, (w_board, h_board), None)
        
        # If found, add object points, image points (after refining them)
        if chessboard_found == True :
            # Refine the corners
            cv2.cornerSubPix(gray_frame, corners, (11, 11), (-1, -1), CRITERIA)
            
            # Get 3x3 homography matrix
            h_matrix = getHomographyMatrix(img_to_augment, corners, w_board, h_board)
            
            # Get warped image
            cam_height, cam_width = undistorted_cam_frame.shape[:2]
            warped_img = cv2.warpPerspective(img_to_augment, h_matrix, (cam_width, cam_height))
            
            # Overlay corners points
            # cv2.drawChessboardCorners(undistorted_cam_frame, (h_board, w_board), corners, chessboard_found)
            
            # Overlay the augmented image   
            warped_mask = np.invert(cv2.threshold(warped_img, 0, 255, cv2.THRESH_BINARY)[1]) / 255
            undistorted_cam_frame = np.multiply(undistorted_cam_frame, warped_mask) + warped_img
            
        cv2.imshow("Augmented reality - Webcam", undistorted_cam_frame)
        if cv2.waitKey(1) & 0xFF == ord('q') : quit()
        
    cam_handler.release()
    cv2.destroyAllWindows()

'''
Get the 4 corners from the whole corners set of the checkerboard

Input parameters:
    board_corners : All corners of the checkerboard
    w_board : Checkerboard width (in tiles)
    h_board : Checkerboard height (in tiles)
    
Output parameters:
    corners_rect : The 4 corners as a rectangle (tl, tr, bl, br)
'''
def getCheckerboardRect(board_corners, w_board, h_board):
    
    # Init rectangle
    corners_rect = np.zeros((4, 2), np.float32)
    
    # Compute each corner point using (tl, tr, bl, br) order
    corners_rect[0] = board_corners[0]
    corners_rect[1] = board_corners[w_board - 1]
    corners_rect[2] = board_corners[w_board*(h_board - 1)]
    corners_rect[3] = board_corners[h_board*w_board - 1]
    
    # Return the rectangle
    return corners_rect

'''
Get the 4 corners from an image

Input parameters:
    image : Source image
    
Output parameters:
    image_rect : The 4 corners as a rectangle (tl, tr, bl, br)
'''
def getImageRect(image):
    
    # Init rectangle
    image_rect = np.zeros((4, 2), np.float32)
    
    # Get image dimensions
    width, height = image.shape[:2]
    
    # Compute each corner point using (tl, tr, bl, br) order
    image_rect[0] = [0, 0]
    image_rect[1] = [width - 1, 0]
    image_rect[2] = [0, height - 1]
    image_rect[3] = [width - 1 - 1, height - 1]
    
    # Return the rectangle
    return image_rect
    
'''
Save camera parameters in a configuration file

Input parameters:
    file_name : The name of the configuration file
    cam_matrix : 3x3 matrix with focal length and optical centers parameters
    dist_coefs : Distortion coefficients given by [k1 k2 p1 p2 p3]
'''
def saveCamParameters(file_name, cam_matrix, dist_coefs):
    np.savez(file_name, cam_matrix, dist_coefs)

'''
Load camera parameters from a configuration file

Input parameters:
    file_name : The name of the configuration file
    
Output parameters:
    parameters : Camera parameters -> cam_matrix, dist_coefs
'''
def loadCamParameters(file_name):
    parameters = np.load(file_name)
    return parameters['arr_0'], parameters['arr_1']
