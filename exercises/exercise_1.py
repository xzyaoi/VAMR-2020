import os.path as osp
import numpy as np
import cv2
from matplotlib import pyplot as plt
from vo_lib.camera import poseVectorToTransformationMatrix, projectPoints, undistortImage, vectorized_undistortImage
DATA_PATH = '../Exercise 1 - Augmented Reality Wireframe Cube/data'

# Load camera poses.
# Each row i of matrix poses contains the transformation that transforms
# Points expressed in the world frame to points expressed in the camera frame.
pose_vectors = np.loadtxt(osp.join(DATA_PATH, 'poses.txt'))

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
img = cv2.imread(osp.join(DATA_PATH, 'images',
                          'img_{0:04d}.jpg'.format(img_index)))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find intersection points
T_C_W = poseVectorToTransformationMatrix(pose_vectors[img_index-1, :])

p_C_corners = np.matmul(T_C_W, np.vstack(
    (p_W_corners, np.ones((1, num_corners)))))
p_C_corners = p_C_corners[0:3, :]

projected_points = projectPoints(p_C_corners, K, D)
'''
plt.imshow(img,cmap = 'gray')
plt.plot(projected_points[0,:], projected_points[1,:],'ro', marker='.',color='r')
plt.show()

# Undistort image without bilinear interpolation
img_undistorted = vectorized_undistortImage(img, K, D)
print(img_undistorted.shape)
plt.imshow(img_undistorted, cmap='gray')
plt.show()
'''
# Draw cube on undistored image
offset_x, offset_y = 0.04 * 3, 0.04
s = 2 * 0.04
x = np.linspace(0, 1, 2)
y = np.linspace(0, 1, 2)
z = np.linspace(-1, 0, 2)
X, Y, Z = np.meshgrid(x,y,z)
p_W_cube = np.vstack((offset_x + X.flatten('F')*s, offset_y+Y.flatten('F')*s, Z.flatten('F')*s))
p_C_cube = np.matmul(T_C_W, np.vstack((p_W_cube, np.ones((1,8)))))
p_C_cube = p_C_cube[0:3,:]

cube_pts = projectPoints(p_C_cube, K)

img_undistorted = vectorized_undistortImage(img, K, D)
plt.imshow(img_undistorted, cmap='gray')

# top layer
plt.plot([cube_pts[0][0], cube_pts[0][1]], [cube_pts[1][0], cube_pts[1][1]], color='r')
plt.plot([cube_pts[0][0], cube_pts[0][2]], [cube_pts[1][0], cube_pts[1][2]], color='r')
plt.plot([cube_pts[0][1], cube_pts[0][3]], [cube_pts[1][1], cube_pts[1][3]], color='r')
plt.plot([cube_pts[0][2], cube_pts[0][3]], [cube_pts[1][2], cube_pts[1][3]], color='r')

# Bottom layer
plt.plot([cube_pts[0][0+4], cube_pts[0][1+4]], [cube_pts[1][0+4], cube_pts[1][1+4]], color='r')
plt.plot([cube_pts[0][0+4], cube_pts[0][2+4]], [cube_pts[1][0+4], cube_pts[1][2+4]], color='r')
plt.plot([cube_pts[0][1+4], cube_pts[0][3+4]], [cube_pts[1][1+4], cube_pts[1][3+4]], color='r')
plt.plot([cube_pts[0][2+4], cube_pts[0][3+4]], [cube_pts[1][2+4], cube_pts[1][3+4]], color='r')

plt.plot([cube_pts[0][0], cube_pts[0][0+4]], [cube_pts[1][0], cube_pts[1][0+4]], color='r')
plt.plot([cube_pts[0][1], cube_pts[0][1+4]], [cube_pts[1][1], cube_pts[1][1+4]], color='r')
plt.plot([cube_pts[0][2], cube_pts[0][2+4]], [cube_pts[1][2], cube_pts[1][2+4]], color='r')
plt.plot([cube_pts[0][3], cube_pts[0][3+4]], [cube_pts[1][3], cube_pts[1][3+4]], color='r')

plt.show()