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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid # 相机的唯一标识符
        self.colmap_id = colmap_id # colmap 算出位姿的 id
        self.R = R # 做投影转换的时候的那个旋转矩阵
        self.T = T # 平移矩阵
        self.FoVx = FoVx # 相机在水平方向上的视野范围（角度）
        self.FoVy = FoVy # 相机在垂直方向上的视野范围（角度）
        self.image_name = image_name # 对应图像的名字

        # 这段代码尝试将 data_device 参数指定的设备转换为 PyTorch 的设备对象，并将结果赋值给 self.data_device。
        # 如果转换过程中出现异常，则捕获异常并输出警告信息，然后将 self.data_device 设置为默认的 CUDA 设备。
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        # 这行代码提取了 resized_image_rgb 的前三个通道作为 gt_image 。
        # 通常情况下，这代表了图像的RGB颜色通道。...用于表示在所有其他维度上选择所有元素。
        # :3 是表示在第一个维度上选择 索引范围从 0 到 3（不包括 3） 的元素。
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None

        # 检查图像的第一个维度大小。
        # 通常图像张量的形状是 (C, H, W)，其中 C 是通道数。
        # 如果 shape[0] == 4，说明图像有 4 个通道，通常代表 Alpha（或掩码）+ RGB 。
        if resized_image_rgb.shape[0] == 4:
            # resized_image_rgb[3:4, ...]: 通过切片提取第 4 个通道（索引为 3），保留其维度（形状变为 (1, H, W)）。
            # .to(self.data_device): 将提取的通道移动到指定的设备（如 GPU 或 CPU），确保计算环境一致。
            # self.alpha_mask: 将提取到的第 4 个通道赋值给 self.alpha_mask，表示它是图像的 alpha 通道或掩码。
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else:
            # torch.ones_like(resized_image_rgb[0:1, ...]): 如果图像不是 4 通道（例如是 3 通道的 RGB 图像），
            # 就生成一个与第一个通道形状相同的全 1 张量，形状为 (1, H, W)，值全为 1。
            # .to(self.data_device): 将生成的张量移动到指定设备。
            # self.alpha_mask: 生成的全 1 张量赋值给 self.alpha_mask，表示默认情况下的 alpha 通道（完全不透明）。
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        # train_test_exp： 判断训练集中是否包含测试集
        # is_test_view： 判断是否为训练数据集
        if train_test_exp and is_test_view:
            # is_test_dataset 一个布尔值，表示当前是否处理测试数据集。
            if is_test_dataset:
                # 选择 self.alpha_mask 最后一个维度（通常是宽度）的前一半区域。
                # :self.alpha_mask.shape[-1] // 2 表示切片到前一半。
                # self.alpha_mask.shape[-1] // 2: 表示从宽度中间到最后的区域。
                # 将 self.alpha_mask 的前半部分设置为 0，表示掩盖（透明或无效）
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                # 选择 self.alpha_mask 最后一个维度的后一半区域。
                # self.alpha_mask.shape[-1] // 2: 表示从宽度中间到最后的区域。
                # 将 self.alpha_mask 的后一半部分设置为 0，表示掩盖（透明或无效）。
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        #这行代码首先对输入的图像数据image进行了clamp操作，将所有像素值限制在0到1之间。
        # 然后，使用to方法将处理后的图像数据移动到指定的设备self.data_device上，并将结果赋值给self.original_image属性。
        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

# 这段代码的功能是：
#   根据输入反深度图和深度参数生成一个缩放、调整并限制范围的反深度图张量。
#   生成相应的深度掩码，标记哪些深度值有效。
#   根据深度参数的合理性判断深度图是否可靠。
        # 设置 self.invdepthmap 为 None，表示默认没有反深度图。
        self.invdepthmap = None
        # 设置 self.depth_reliable 为 False，表示默认深度图不可用或不可靠。
        self.depth_reliable = False
        # 检查输入条件：确保反深度图 invdepthmap 和深度参数 depth_params 非空。确保深度参数中的 scale（缩放系数）大于 0。
        if invdepthmap is not None and depth_params is not None and depth_params["scale"] > 0:
            # 深度图缩放与偏移：对输入的反深度图进行线性变换 invdepthmapScaled = invdepthmap × scale + offset。
            # scale 用于调整深度图的比例，offset 用于调整深度图的基准值。
            invdepthmapScaled = invdepthmap * depth_params["scale"] + depth_params["offset"]
            # 调整分辨率： 使用 OpenCV 的 cv2.resize 函数将反深度图调整到目标分辨率 resolution。
            invdepthmapScaled = cv2.resize(invdepthmapScaled, resolution)
            # 处理非法值：将反深度图中小于 0 的值设置为 0，确保深度图的值为非负。
            invdepthmapScaled[invdepthmapScaled < 0] = 0

            # 确保深度图是二维数据，并转换为 PyTorch 张量，放置在指定设备上。
            # 确保深度图维度：检查深度图是否是二维的（高度和宽度）。
            if invdepthmapScaled.ndim != 2:
                # 如果是三维或多维数组，仅提取第一通道。
                invdepthmapScaled = invdepthmapScaled[..., 0]
            # 转换为张量：将处理后的反深度图 invdepthmapScaled 转换为 PyTorch 张量。
            # 添加一个新维度（None 表示批次维度）。将张量移动到指定的设备（self.data_device）。
            self.invdepthmap = torch.from_numpy(invdepthmapScaled[None]).to(self.data_device)

            # 使用 alpha_mask 或生成默认掩码作为深度掩码。
            # 生成深度掩码：
            #   如果存在 alpha_mask，使用其克隆值作为深度掩码。
            #   否则，生成一个与 invdepthmap 形状相同的掩码，其中值为 1（深度有效时）。
            if self.alpha_mask is not None:
                self.depth_mask = self.alpha_mask.clone()
            else:
                self.depth_mask = torch.ones_like(self.invdepthmap > 0)

            # 如果缩放系数过小或过大，则将深度掩码设置为全零。否则，标记深度图为可靠。
            # 检查深度可靠性：
            #   如果 scale 超出与 med_scale（深度中值比例）相关的合理范围，则设置掩码为 0（所有深度无效）。
            if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]: 
                self.depth_mask *= 0
            # 否则，将 self.depth_reliable 设置为 True，表示深度图是可靠的。
            else:
                self.depth_reliable = True

        # 这两行设置了相机视角的远点和近点。
        self.zfar = 100.0
        self.znear = 0.01

        # 这两行分别将平移向量和缩放比例赋值给了对象的属性。
        self.trans = trans
        self.scale = scale

        # 这行代码计算了世界坐标到相机视图坐标的变换矩阵，并将其转置后转移到GPU上。
        # 它调用了名为 getWorld2View2 的函数，该函数接受了相机的旋转矩阵 R、平移向量 T、相机中心平移变换 trans 和缩放变换 scale，
        # 然后返回了一个相机到世界视图的变换矩阵，最后将其转置后转移到GPU上。
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()

        # 这行代码计算了投影矩阵，将高斯点转到一个归一化的一个坐标系里面，并将其转置后转移到GPU上。
        # 具体来说，它调用了名为 getProjectionMatrix 的函数，该函数接受了近裁剪面、远裁剪面、水平和垂直视场角，
        # 然后返回了一个投影矩阵，最后将其转置后转移到GPU上。
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()

        # 这行代码计算了世界到投影坐标的完整变换矩阵。
        # 首先，通过 unsqueeze(0) 将 world_view_transform 和 projection_matrix 分别添加了一个维度，
        # 然后使用 bmm 函数进行矩阵相乘，得到了世界到NDC归一化投影坐标的变换矩阵，最后使用 squeeze(0) 去除了添加的维度。
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        # 这行代码计算了相机在世界坐标系中的中心位置。首先，使用 inverse() 方法求得 world_view_transform 的逆矩阵，
        # 然后取其第四行的前三个元素，即相机中心在世界坐标系中的位置。
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

