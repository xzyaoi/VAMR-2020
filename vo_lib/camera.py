import numpy as np


def poseVectorToTransformationMatrix(pose_vec):
    omega = pose_vec[0:3]
    t = pose_vec[3:6]
    theta = np.linalg.norm(omega)
    k = omega/theta
    kx, ky, kz = k[0], k[1], k[2]
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    R = np.identity(3) + np.sin(theta) * K + (1-np.cos(theta)) * K * K
    T = np.identity(4)
    T[0:3,0:3] = R
    T[0:3, 3] = t
    return T