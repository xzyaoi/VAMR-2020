import os.path as osp
import numpy as np
import cv2
from vo_lib.camera import poseVectorToTransformationMatrix
DATA_PATH = '../Exercise 1 - Augmented Reality Wireframe Cube/data'

# Load camera poses.
## Each row i of matrix poses contains the transformation that transforms 
## Points expressed in the world frame to points expressed in the camera frame.
pose_vectors = np.loadtxt(osp.join(DATA_PATH,'poses.txt'))

# Define 3D corner positions
square_size = 0.04
num_corners_x, num_corners_y = 9, 6
num_corners = num_corners_x * num_corners_y

x = np.linspace(0, num_corners_x-1, num_corners_x)
y = np.linspace(0, num_corners_y-1, num_corners_y)
X, Y = np.meshgrid(x, y)
c = np.vstack((X.flatten('F'), Y.flatten('F')))
p_W_corners = square_size * np.vstack((X.flatten('F'), Y.flatten('F'))).T
p_W_corners = np.hstack((p_W_corners, np.zeros((num_corners, 1)))).T

# Load camera intrinsics

K = np.loadtxt(osp.join(DATA_PATH, 'K.txt'))
D = np.loadtxt(osp.join(DATA_PATH, 'D.txt'))

img_index = 1
img = cv2.imread(osp.join(DATA_PATH, 'images','img_{0:04d}.jpg'.format(img_index)))

T_C_W = poseVectorToTransformationMatrix(pose_vectors[img_index,:])
print(T_C_W)