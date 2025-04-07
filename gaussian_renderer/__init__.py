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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建零张量。我们将使用它来制作2D（屏幕空间）方式的pytorch返回梯度
    # 这行代码首先调用了 torch.zeros_like() 函数，该函数会创建一个与 pc.get_xyz 张量相同形状和数据类型的全零张量。
    # 然后，通过设置 requires_grad=True，表示我们希望 PyTorch 跟踪并计算这个张量的梯度。
    # 最后，通过 device="cuda" 将张量放置在 CUDA 设备上。+ 0 的操作不会改变张量的值，只是确保其梯度信息被正确地追踪。
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        # 这行代码尝试保留 screenspace_points 张量的梯度信息。在 PyTorch 中，当张量执行反向传播时，梯度信息会被默认清除，
        # 但通过调用 retain_grad() 方法可以保留梯度信息。这样做是为了在需要时能够访问和使用张量的梯度信息。
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # 这行代码计算了视角水平方向上一半视场角的切线值。
    # viewpoint_camera.FoVx 是视角的水平视场角（Field of View），乘以 0.5 是因为切线值是基于一半视场角的。
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # 这行代码计算了视角垂直方向上一半视场角的切线值。
    # viewpoint_camera.FoVy 是视角的垂直视场角，乘以 0.5 是因为切线值是基于一半视场角的。
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform, # 设置视图矩阵，这个矩阵通常描述了世界坐标系到观察坐标系的变换，用于将场景坐标转换为相机视图空间。
        projmatrix=viewpoint_camera.full_proj_transform, # 设置投影矩阵，这个矩阵通常描述了投影变换，用于将相机视图空间中的坐标投影到屏幕空间。
        sh_degree=pc.active_sh_degree, # 设置球谐函数的度数，这个参数通常与光照或反射模型相关。
        campos=viewpoint_camera.camera_center, # 设置相机位置，通常是相机的位置坐标。
        prefiltered=False,
        debug=pipe.debug
    )

    # 这个 rasterizer 对象可能是用于执行高斯光栅化的工具，它使用了之前设置好的光栅化参数来进行渲染。
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz # (224668,3)
    means2D = screenspace_points # (224668,3)
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 如果提供了预先计算的三维协方差，请使用它。如果没有，则光栅化器将根据缩放/旋转来计算它。
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # 这段代码似乎用于处理点云数据的颜色计算，基于球谐函数（SH）
    shs = None
    colors_precomp = None
    # 如果未提供 override_color（即为 None），则代码检查 pipe.convert_SHs_python 是否为 True。
    # 如果是，它会使用球谐函数计算颜色。
    if override_color is None:
        if pipe.convert_SHs_python:
            # pc.get_features 是一个张量，其形状为 (num_points, num_channels, batch_size)。(224668,16,3)。
            # transpose(1, 2) 将特征张量的第二和第三个维度进行转置，即将 num_channels 和 batch_size 互换位置。(224668,3,16)。
            # view(-1, 3, (pc.max_sh_degree+1)**2) 对转置后的张量进行重新形状操作，
            # 其中 -1 表示根据其他维度的大小自动推断出该维度的大小 ,3 表示每个点的特征维度为3，
            # 而 (pc.max_sh_degree+1)**2 则是球谐函数的维度。(224668,3,16)。
            # 最终得到的 shs_view 是一个三维张量，其形状为 (num_points, 3, (pc.max_sh_degree+1)**2)，
            # 每个元素对应一个点的特征，其中每个点的特征维度为3，表示点的颜色信息，球谐函数的维度表示颜色的分量。
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # 这行代码计算了每个点到视角相机中心的方向向量 dir_pp。
            # pc.get_xyz 似乎是一个张量，表示点云中每个点的空间坐标。它的形状是 (num_points, 3)。(224668,3)
            # viewpoint_camera.camera_center 是该视角相机的中心坐标。 形狀是(3)
            # .repeat(pc.get_features.shape[0], 1) 用于将 viewpoint_camera.camera_center 在第一个维度（batch_size 维度）上复制，以与 pc.get_xyz 的批量大小相匹配。(224668,3)
            # 执行减法操作，即 pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)，得到每个点到相机中心的位移向量。
            # 最终，dir_pp 是一个张量，形状与 pc.get_xyz 相同，表示了每个点到相机中心的方向向量。
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # 这行代码计算了每个点到相机中心的方向向量 dir_pp 的归一化版本 dir_pp_normalized。
            # dir_pp 是一个张量，表示每个点到相机中心的方向向量。它的形状应该与点云的形状相同，即 (num_points, 3)，num_points 是点的数量，3 是每个点的三维坐标。
            # dir_pp.norm(dim=1, keepdim=True) 计算了 dir_pp 沿着第一个维度（即点的维度）的范数（即每个向量的长度），并保持结果的维度不变。
            # dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True) 将每个点的方向向量 dir_pp 分别除以对应的长度，这样做就将每个向量归一化为单位长度，从而得到了归一化后的方向向量 dir_pp_normalized。
            # dir_pp_normalized 是一个与 dir_pp 相同形状的张量，其中每个点的方向向量都被归一化为单位长度。
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # 函数 eval_sh 可能的作用是根据球谐函数系数和方向向量计算颜色值。
            # 具体来说，它可能接受球谐函数的度数、球谐函数系数和方向向量作为输入，并返回对应于每个点的颜色值。
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # 这行代码用于对计算得到的颜色值进行调整，确保其值不低于0。
            # sh2rgb 是一个张量，表示从球谐函数和方向向量计算得到的颜色值。它的形状可能与 shs_view 相同，即 (num_points, 3, 16)。(224668,3,16)
            # 0.5 被添加到 sh2rgb 中，这可能是为了调整颜色值的范围。
            # torch.clamp_min(input, min) 是 PyTorch 中的一个函数，它将输入张量的每个元素限制在最小值 min 以上。
            # 在这里，torch.clamp_min(sh2rgb + 0.5, 0.0) 的作用是将颜色值的最小值限制在0以上。
            # 最终，colors_precomp 被设置为经过调整后的颜色值张量。
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        # 如果 pipe.convert_SHs_python 不为 True，则将 shs 赋值为 pc.get_features。
        else:
            shs = pc.get_features
    else:
        # 如果提供了 override_color，则将 colors_precomp 设置为 override_color。
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # 光栅化可见的高斯图像，获得它们的半径（在屏幕上）。
    rendered_image, radii, depth_image = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    # 在训练阶段，将曝光参数应用到渲染图像上。
    if use_trained_exp:
        # 调用 pc.get_exposure_from_name(viewpoint_camera.image_name)
        # 获取与当前视点摄像机名称 viewpoint_camera.image_name 对应的曝光矩阵。
        # exposure 应该是一个形状为 3×4 的张量，包含颜色调整和亮度偏移信息。
        # 根据当前视点摄像机名称 viewpoint_camera.image_name 获取曝光矩阵。
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        # 转换维度：rendered_image.permute(1, 2, 0) 将渲染图像的维度从C*H*W 转为 H*W*C ,方便矩阵乘法操作。
        # 颜色调整：使用 torch.matmul 将图像像素与曝光矩阵的前三列 exposure[:3, :3] 相乘，调整颜色。
        # 恢复维度：.permute(2, 0, 1) 将图像维度恢复为 C*H*W。
        # 亮度偏移：添加曝光矩阵的最后一列 exposure[:3, 3]，并扩展到图像大小（通过 None, None 实现）。
        # 调整渲染图像的颜色和亮度：
        #               使用曝光矩阵的前三列调整颜色；
        #               使用曝光矩阵的最后一列调整亮度。
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # 那些截锥体被剔除或半径为0的高斯不可见。
    # They will be excluded from value updates used in the splitting criteria.
    # 它们将被排除在拆分条件中使用的值更新之外。
    # clamp 函数：
    #   PyTorch 的 clamp 函数用于将张量的值限制在指定范围内。
    #   clamp(0, 1) 将 rendered_image 中的所有值限制在 [0, 1] 区间：
    #       如果某个值小于 0，则将其设置为 0。
    #       如果某个值大于 1，则将其设置为 1。
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out
