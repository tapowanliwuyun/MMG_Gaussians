#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

# 世界坐标系到相机坐标系的一个转换
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    # RT矩阵就是世界坐标系到相机坐标系的一个转换矩阵
    # 但是这个投影矩阵是有问题的，这个投影矩阵是由世界坐标系到相机坐标系的一个投影矩阵，
    # 就是相机中心实际上是有变化的，用translate数组表达了出来，就是我们的相机中心不是原来的相机中心了，他要加上这个转换后的这个相机中心。
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose() # 转置
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    # 那怎么把这个转换加进来呢：首先对RT求一个逆，此时RT的逆矩阵的 [0,3],[1,3],[2,3]三维维度就是相机中心的位置，
    # 当前矩阵就是由相机到世界坐标系的一个转换矩阵
    # 然后提取中心，，加上变化以及尺度变化，最后在赋值回去，再求逆求回来 。

    # 使用 np.linalg.inv() 函数计算 Rt 的逆矩阵，得到从相机视图坐标系到世界坐标系的变换矩阵 C2W。
    C2W = np.linalg.inv(Rt)
    # 从变换矩阵 C2W 中提取相机中心的位置。
    cam_center = C2W[:3, 3]
    # 将平移变换 translate 加到相机中心位置，并乘以缩放变换 scale，得到新的相机中心位置。
    cam_center = (cam_center + translate) * scale
    # 将新的相机中心位置赋值给变换矩阵 C2W 的前三行的第四列。
    C2W[:3, 3] = cam_center
    # 使用 np.linalg.inv() 函数计算新的变换矩阵 C2W 的逆矩阵，得到从世界坐标系到相机视图坐标系的变换矩阵。
    Rt = np.linalg.inv(C2W)
    # 将变换矩阵转换为 np.float32 类型，并返回。
    return np.float32(Rt)

    # 这个函数用于计算投影矩阵，该矩阵用于将视景体中的点投影到裁剪空间。
def getProjectionMatrix(znear, zfar, fovX, fovY):
    # 计算水平和垂直方向的视场角的一半的正切值。这里使用了 math.tan() 函数来计算角度的正切值。
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    # 根据视场角的一半的正切值和近裁剪面的距离，计算裁剪空间的上下左右边界。
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    # 创建一个 4x4 的零矩阵 P，用于存储投影矩阵。
    P = torch.zeros(4, 4)

    z_sign = 1.0

    # 此处介绍一下 可以查看 https://www.bilibili.com/video/BV1UT421r73g?t=3433.5
    # 用P 乘 相机坐标系下的点 会得到NDC归一化坐标系中的点，
    # 归一化坐标系下的点的意义在于：
    # 1. 对于图像，采集的相机不一样，相机参数也不同，相机视角也不一样，使用归一化坐标可以消除不同相机之间的影响
    # 为后面 将NDC坐标系下的带你 统一投影到像素平面  做准备
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))