import numpy as np


def poseVectorToTransformationMatrix(pose_vec):
    '''
    Converts a 6x1 pose vector into a 4x4 transformation matrix.
    Args:
        pose_vec: pose vector
    '''
    omega = pose_vec[0:3]
    t = pose_vec[3:6]
    theta = np.linalg.norm(omega)
    k = omega/theta
    kx, ky, kz = k[0], k[1], k[2]
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    R = np.identity(3) + np.sin(theta) * K + \
        (1-np.cos(theta)) * np.linalg.matrix_power(K, 2)
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T


def distortPoints(x, D, K):
    '''
    Applies lens distribution D (2x1) to 2D points x (2xN) on the image plane.
    Args:
        xL 2D points
        D: Distortion matrix
        K: Camera matrix
    '''
    k_1, k_2 = D[0], D[1]
    u_0 = K[0, 2]
    v_0 = K[1, 2]
    xp, yp = x[0, :] - u_0, x[1, :]-v_0
    r2 = xp * xp + yp * yp
    xpp = u_0 + xp * (1+k_1*r2 + k_2 * r2*r2)
    ypp = v_0 + yp * (1+k_1*r2 + k_2 * r2*r2)
    x_d = np.vstack((xpp, ypp))
    return x_d


def projectPoints(points_3d, K, D=np.zeros((4, 1))):
    '''
    Projects 3d points to the image plane (3xN), given the camera matrix K (3x3) and the distortion coefficients D (4x1).
    If distortion is not given, assume zero distortion
    '''
    # get image coordinates
    projected_points = np.matmul(K, points_3d)
    projected_points = projected_points / projected_points[2, :]

    # apply distortion
    projected_points = distortPoints(projected_points[0:2, :], D, K)
    return projected_points


def undistortImage(img, K, D, bilinear_interpolation=0):
    '''
    Corrects an image from lens distortion
    '''
    height, width = img.shape
    undistorted_img = np.zeros((height, width), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            x_d = distortPoints(np.array([[x-1], [y-1]]), D, K)
            u, v = x_d[0], x_d[1]
            # Bilinear interpolation
            u_1, v_1 = int(np.floor(u)), int(np.floor(v))
            if bilinear_interpolation > 0:
                a, b = u - u_1, v-v_1
                if u_1 + 1 > 0 and u_1+1 <= width and v_1 + 1 > 0 and v_1+1 <= height:
                    first_term = (1-b) * \
                        ((1-a)*img[v_1, u_1]+a*img[v_1, u_1+1])
                    second_term = b * \
                        ((1-a)*img[v_1+1, u_1] + a*img[v_1+1, u_1+1])
                    undistorted_img[y, x] = first_term + second_term
                else:
                    if u_1 > 0 and u_1 <= width and v_1 > 0 and v_1 <= height:
                        undistorted_img[y, x] = img[v_1, u_1]
    return undistorted_img


def vectorized_undistortImage(img, K, D):
    x = np.linspace(0, img.shape[1]-1, img.shape[1])
    y = np.linspace(0, img.shape[0]-1, img.shape[0])
    X, Y = np.meshgrid(x, y)
    px_locs = np.vstack((X.flatten('F'), Y.flatten('F')))
    dist_px_locs = distortPoints(px_locs, D, K)
    c = (np.rint(dist_px_locs[1,:])+img.shape[0] * np.rint(dist_px_locs[0,:])).astype(np.int)
    print(c)
    intensity_vals = img.flatten(order='F')[c]
    print(intensity_vals)
    undistorted_img = np.reshape(intensity_vals, img.shape, order='F').astype(np.uint8)
    return undistorted_img