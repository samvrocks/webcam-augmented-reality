'''
Webcam Augmented Reality
@author: Samuel Veloso - samvrocks
'''

import camera as cam
import os.path
import cv2

''' INITIALIZATION PARAMETERS '''
w_board, h_board = (6, 9) # checkboard dimensions


''' LOAD OR GENERATE CAM PARAMETERS '''
cam_parameters_config = 'params.npz'

# Get parameters from file configuration if exists
if os.path.isfile(cam_parameters_config):  
    cam_matrix, dist_coefs = cam.loadCamParameters(cam_parameters_config)
# Generate parameters from webcam if file doesn't exist
else: 
    rms, cam_matrix, dist_coefs, r_vecs, t_vecs = cam.calibrateWebcam(w_board, h_board, 10)
    cam.saveCamParameters(cam_parameters_config, cam_matrix, dist_coefs)

#print "Camera matrix:\n", cam_matrix
#print "Distortion coefficients:\n", dist_coefs.ravel()

''' PLANAR BASED AUGMENTED REALITY '''
img_to_augment_name = raw_input('Introduce the name of the image: ')
if os.path.isfile(img_to_augment_name):  
    img_to_augment = cv2.imread(img_to_augment_name)
else:
    print 'Image not found'
    quit()

print "\nReality is being augmented... Press 'q' key to stop."
cam.playAugmentedReality(img_to_augment, cam_matrix, dist_coefs, w_board, h_board)
