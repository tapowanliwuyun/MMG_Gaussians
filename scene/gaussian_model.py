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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree):
        self.active_sh_degree = 0 # 当前球谐函数的阶数。
        self.max_sh_degree = sh_degree # 被设置为传入的 sh_degree 参数的值，这是一个表示最大球谐度的属性。
        self._xyz = torch.empty(0) # 椭球位置
        self._features_dc = torch.empty(0) # 球谐函数的直流分量
        self._features_rest = torch.empty(0) # 球谐函数的高阶分量
        self._scaling = torch.empty(0) # 缩放因子
        self._rotation = torch.empty(0) # 旋转因子
        self._opacity = torch.empty(0) # 不透明度
        self.max_radii2D = torch.empty(0) # 这是投影到平面上的二维高斯分布的最大半径
        self.xyz_gradient_accum = torch.empty(0) # 点云位置的梯度的累积值
        self.denom = torch.empty(0) # 统计的分母的数量，最后累积的位置梯度要除以这个的
        self.optimizer = None # 初始化为 None，表示优化器尚未设置，训练的时候用到的。
        self.percent_dense = 0 # 百分比密度，做密度控制的时候用的
        self.spatial_lr_scale = 0 # 学习率的因子
        self.setup_functions() # 最后，调用了 self.setup_functions() 方法，设置激活函数的方法。

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    # 迭代球谐函数的阶数，只要当前阶数小于设置的最大阶数，运行该函数就会增加迭代
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        # 学习率的变化因子
        # spatial_lr_scale 参数设置了空间分辨率的尺度，并将其保存到模型属性 self.spatial_lr_scale 中。
        self.spatial_lr_scale = spatial_lr_scale

        # 其中 fused_point_cloud 是点云的位置信息。
        # 创建一个张量保存点云数据，把数组类型的点云数据存到该当量里面
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # RGB2SH函数 将RGB的颜色信息 转换为SH球谐函数的系数。
        # fused_color 是点云的颜色信息。
        # fused_color 当前只存了零阶直流分量球谐函数的值，后面才加上更高阶的分量
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # 创建一个用于存储点云颜色特征的张量 features，初始化为零。
        # 该张量的形状为 (高斯分布总数量, 3（就是三个通道）, (最大球形谐波阶数 + 1) ** 2)，其中每个点云特征都是一个球形谐波系数。
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # 将高斯球的颜色信息中的球谐第0阶直流分量存储到特征张量的第一个通道（对应于球形谐波系数的零阶）中。
        features[:, :3, 0 ] = fused_color
        # 其余更高阶部分设置为零。
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # 首先，将点云的位置信息转换为 PyTorch 张量，并将其移动到 GPU 上。
        # 然后，调用 simple_knn 的 distCUDA2 函数，计算点云中的每个点到与其最近的K个点的平均距离的平方，
        # 可以在 submodules/simple-knn/spatial.cu 查看函数
        # 并使用 torch.clamp_min 函数将距离限制在一个最小值 0.0000001 以上，以防止出现零距离的情况。
        # 就是两个高斯椭球一定不会重合，一定会有最小距离。
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # 接着，将距离平方开平方并取对数，得到每个点到原点的欧氏距离。
        # 然后，将其重复三次，以便与三维坐标对应。这样就得到了每个点的缩放因子，用于调整点云的大小。
        # 对于一个高斯点，去检索这个高斯点最近的三个高斯点，然后把这三个高斯点的距离，取一个平均值，用这个平均值去构建一个高斯球，
        # 这个就是使用点云构建的初始高斯球。这个高斯球的半径就是 scales 这个量
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # 首先，创建了一个形状为 (高斯球数量, 4) 的零张量，其中每个点的旋转表示为一个四元数。
        # 这里使用了 torch.zeros() 函数来创建零张量，并将其移动到 GPU 上。
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        # 接着，将 rots 张量的第一列（四元数的实部）设置为 1，以表示单位四元数。
        # 这样就初始化了一个单位四元数张量，用于表示点云的旋转。
        rots[:, 0] = 1

        # 首先，创建了一个形状为 (点云数量, 1) 的张量，其中每个点的不透明度表示为一个标量。
        # 然后，将该张量的每个元素初始化为 0.1。
        # 最后，使用反激活函数 sigmoid 函数对这些值进行变换，这是因为存透明度的时候使用激活函数去存，
        # 所以取的时候使用反激活函数去取。这样就得到了一个用于表示点云不透明度的张量 opacities。
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 将点云的位置信息作为模型的参数 _xyz，并将其标记为可训练参数。
        # 这里使用了 nn.Parameter 将张量转换为模型的可训练参数，
        # 并使用 requires_grad_(True) 方法启用梯度优化。
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # 将点云的颜色信息的零阶球形谐波系数作为模型的参数 _features_dc，并将其标记为可训练参数。
        # contiguous()用来保证其是连续的，并且也是梯度优化的。
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # 将点云的颜色信息的非零阶球形谐波系数作为模型的参数 _features_rest，并将其标记为可训练参数。
        # 同样，在这之前，对颜色信息进行了处理，去除了零阶球形谐波系数，并将其转置和连续化以满足 PyTorch 的要求。
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # 将点云的缩放因子作为模型的参数 _scaling，并将其标记为可训练参数。
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        # 将点云的旋转信息作为模型的参数 _rotation，并将其标记为可训练参数。
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        # 将点云的不透明度信息作为模型的参数 _opacity，并将其标记为可训练参数。
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # 创建一个用于存储每个高斯二维投影后的高斯分布的最大半径的张量，并将其初始化为零。
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        # 给每个点初始化一个梯度累积
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # denom 用来以后计算平均梯度的，用累积梯度除
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 这里用 l 表示的是所有需要优化的项的学习率,因为学习率是动态的，就是训练过程中的学习率不是统一定好的
        # 注意到零阶球谐系数分量 features_dc 被拿出来单独学习,而其他分量学习率除以了20
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # 'params'：这个键可能保存了在训练过程中将要更新的参数列表。
            # 在你的例子中，它包含了 [self._xyz] ，这表明 _xyz 是一个将要被更新的参数。
            # 'lr'：这个键可能代表了优化算法的学习率。看起来它是根据一些初始学习率（training_args.position_lr_init）
            # 乘以一个空间学习率缩放因子（self.spatial_lr_scale）计算得出的。
            # 'name'：这个键似乎是一种用来标识参数集的标签或标识符。在你的例子中，它是"xyz"
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # 然后再把我们的优化器设置为深度学习里面用的这个 Adam 优化器，设置初始学习率为0
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        if self.pretrained_exposures is None:
            self.exposure_optimizer = torch.optim.Adam([self._exposure])

        # 虽然没看到这东西从哪来的，但是这个是用来指数调节学习率的
        # 我们使用标准的指数衰减调度技术，类似于Plenoxels [Fridovich-Keil和Yu等人2022]，但仅用于位置。
        # 这就相当于是定义了一些函数的别名，这里返回的是一个函数，根据输入的步数，返回最终的权重值。
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            # 更新学习率的时候值更新了位置 xyz 的学习率,通过 training_setup 里面搞的 scheduler_args 算了新的 lr
            # 如果你的参数识别到是 xyz ，那你就开始对这个学习率进行优化，
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                # 把优化后的学习率传到这个变量里面
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # 重置不透明度的操作
    def reset_opacity(self):
        # 把不透明度设置成 0.01 和原不透明度之间的小值，然后用 inverse_sigmoid 激活一下
        # 这就是原论中所说的一定迭代次数内要重置
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        # 把新的不透明度添加到优化器里面
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        # 转成张量的形式，赋到这个对象里面
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    # 就是把张量转到这个优化器里面用于优化，前面用张量存储数据，但是迭代的时候要用优化器去进行迭代训练
    # 用变量替换到优化器里面去
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                # 深度学习里面的概念 动量 也就是梯度下降的方向
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                # 二次动量
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                # 如果变量发生变化，就是张量在训练过程中发生了变化，但是优化器里面存储的状态，也就是优化的状态是不能变的，
                # 这样才能保证你的参数在优化的过程中是一个平滑的迭代的过程。
                # 比如现在有一个参数发生了一个替换，或者说发生了一个比较大的变化，那优化器了里面存的这个状态，就是这个梯度动量，他们应该是不变的
                # 应该取保证一个原有的一个状态，这样原有的优化状态就不会丢失，最后损失函数是一个平滑的下降状态。
                # 参考 https://www.bilibili.com/video/BV1UT421r73g?t=1851.0

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # 重置优化器
    # 只选择保留你需要保留的状态，设定了一个掩码，想保留哪个就用掩码去保留，其他的都会重置为0
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # 剔除
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    # 创建新的张量把它存到优化器里面，添加新的高斯点所需要优化的变量
    # 它的作用是将额外的张量（extension_tensor）添加到优化器的参数中，并更新优化器的状态。
    def cat_tensors_to_optimizer(self, tensors_dict):
        # 初始化一个空字典，用于存储添加到优化器中的张量。
        optimizable_tensors = {}
        # 遍历优化器的参数组。
        for group in self.optimizer.param_groups:
            # 确保每个参数组中只有一个参数。这是一个断言，如果不符合条件会抛出异常。
            assert len(group["params"]) == 1
            # 从 tensors_dict 字典中获取名为 group["name"] 的额外张量。
            extension_tensor = tensors_dict[group["name"]]
            # 获取当前参数组的优化器状态，如果没有则返回 None。
            # 这行代码是在尝试从优化器的状态字典中获取当前参数组的状态信息，其中键为当前参数组的参数。
            # self.optimizer.state 是一个字典，其中包含了优化器的状态信息。
            # 这个字典通常由优化器自动管理， 它会在训练过程中记录参数的状态， 比如梯度的指数加权平均值等。
            # group['params'][0] 表示当前参数组的参数，它通常是一个张量或参数（Parameter）对象。
            # 在 PyTorch 中，参数组是一个字典，其中包含了当前参数组的各种信息，比如参数张量、学习率等。
            # self.optimizer.state.get(..., None) 使用了字典的 get() 方法，它尝试从字典中获取指定键对应的值。
            # 如果键存在，则返回该键对应的值；如果键不存在，则返回 None。
            stored_state = self.optimizer.state.get(group['params'][0], None)
            # 检查当前参数组是否有存储的状态。
            if stored_state is not None:
                # 这行代码的作用是将当前参数组存储的梯度指数加权平均值（exp_avg）与一个与额外张量（extension_tensor）相同形状的全零张量进行拼接，
                # 然后将结果重新赋值给 exp_avg。
                # torch.zeros_like(extension_tensor)：这部分代码创建了一个与 extension_tensor 相同形状的全零张量。
                # torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)：这部分代码使用 torch.cat() 函数沿着
                # 维度 0(一般是样本数的那个维度) 将存储的梯度指数加权平均值和全零张量进行拼接，
                # 这样就把全零张量添加到了梯度指数加权平均值的末尾。
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                # 将新的张量与存储的状态中的指数加权平均和平方指数加权平均拼接起来。
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                # 删除当前参数组的旧状态。
                del self.optimizer.state[group['params'][0]]
                # 将当前参数组的参数与额外张量拼接起来，并将结果设置为一个可优化的参数。
                # torch.cat((group["params"][0], extension_tensor), dim=0)：这部分代码使用 torch.cat() 函数沿着维度0将当前参数组的参数和额外张量进行拼接，得到一个新的张量。
                # nn.Parameter(...)：这部分代码使用 nn.Parameter() 将拼接后的张量重新包装成一个可优化的参数，这样它就可以被优化器进行管理和更新了。
                # .requires_grad_(True)：这部分代码将新创建的参数设置为可求导（requires_grad）的，这样它可以参与反向传播和梯度计算，从而在训练过程中更新梯度。
                # group["params"][0] = ...：这部分代码将新创建的可优化参数重新赋值给当前参数组，更新了参数组中的参数。
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                # 更新优化器的状态。
                self.optimizer.state[group['params'][0]] = stored_state

                # 将添加到优化器的张量添加到 optimizable_tensors 字典中，键是参数组的名称，值是参数。
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        # 将d中的优化变量添加到优化器中，完成新的高斯的创建
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # 这段代码是一个函数 densify_and_split，它的作用是根据一定条件对点云数据进行稠密化和分割处理。
    # 传入参数：目前优化过程中的梯度、梯度阈值、场景范围、一个特定常数也就是分裂为几个
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        # 获取初始点云数据的点数。
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        # 创建一个形状为 (n_init_points,) 的全零张量，并将其放在 GPU 上，大小就是高斯分布总数。
        padded_grad = torch.zeros((n_init_points), device="cuda")
        # 将输入的梯度张量 grads 填充到 padded_grad 张量的前面一部分。
        # squeeze() 方法用于移除 grads 中形状为 1 的维度。
        padded_grad[:grads.shape[0]] = grads.squeeze()
        # 根据梯度阈值，创建一个布尔掩码，表示哪些点被选中。
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # 进一步筛选掩码，以确保选中的点满足密度要求。 就是高斯分布的缩放因子中最大的一个维度的值大于这个场景的范围*对应的比例因子。
        # 也就是将其设置为true，要进行分裂操作
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        # 根据选中的点获取标准差，并复制为 N 组。
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        # 创建一个形状为 (stds.size(0), 3) 的全零张量作为均值。
        means =torch.zeros((stds.size(0), 3),device="cuda")
        # 根据 均值 和 标准差 生成正态分布的样本。
        # torch.normal(mean=means, std=stds)：这部分代码调用了 PyTorch 中的 torch.normal() 函数，用于从正态分布中生成随机样本。
        # 参数 mean 是一个张量，表示样本的均值，而 std 是一个张量，表示样本的标准差。这两个张量的形状必须能够进行广播，以便正确地计算每个维度上的随机样本。
        # 在这里，means 张量是形状为 (num_points, 3) 的全零张量，用于表示样本的均值。而 stds 张量是形状为 (num_points, 3) 的张量，包含了每个点的标准差信息。
        # 因为 means 是全零张量，所以实际上就是从以零为中心的正态分布中生成了样本。 torch.normal() 函数会根据均值和标准差生成随机样本，
        # 使得这些样本在以给定均值为中心、给定标准差为方差的正态分布中分布。
        samples = torch.normal(mean=means, std=stds)
        # 根据选中的点的旋转信息构建旋转矩阵，并复制为 N 组。
        # build_rotation(self._rotation[selected_pts_mask])：这部分代码调用了 build_rotation 函数，它接受一个旋转信息的张量，
        # 并返回对应的旋转矩阵。selected_pts_mask 是一个布尔掩码，用于选择满足条件的点的旋转信息。
        # .repeat(N, 1, 1)：这部分代码将旋转矩阵张量在维度0上重复N次，维度1和维度2上不进行重复。这样做的结果是将选中的旋转矩阵复制为N份，
        # 以便后续的处理。这种操作在扩展张量维度时很常见，可以用于数据的扩展和复制。
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)

        #  将正态分布的样本根据旋转矩阵进行变换，并加上原始点云的坐标。
        # torch.bmm(rots, samples.unsqueeze(-1))：这部分代码调用了 torch.bmm() 函数，该函数用于执行批量矩阵乘法。
        # rots 是形状为 (num_points, 3, 3) 的旋转矩阵张量，samples.unsqueeze(-1) 是形状为 (num_points, 3, 1) 的张量，通过在最后一个维度上增加一个维度，
        # 将样本的形状转换为 (num_points, 3, 1)，表示每个样本是一个列向量。torch.bmm() 函数将这两个张量进行矩阵乘法，得到形状为 (num_points, 3, 1) 的张量。
        # rots 就是旋转，samples就是缩放因子，因此两者相乘就是协方差矩阵。然后加上原本的高斯分布的位置，就是更新后的高斯分布的位置
        # .squeeze(-1)：这部分代码调用了 squeeze() 函数，将结果张量的最后一个维度压缩，即将形状为 (num_points, 3, 1) 的张量转换为
        # 形状为 (num_points, 3) 的张量。
        # + self.get_xyz[selected_pts_mask].repeat(N, 1)：这部分代码将原始点云的坐标信息根据 selected_pts_mask 进行筛选，
        # 并将选中的点的坐标信息复制为N份。然后将上面计算得到的旋转后的样本坐标与原始点云的坐标相加，得到新的坐标。
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # 根据选中的点的缩放信息生成新的缩放值。
        # 这个获取缩放因子之后，除以了0.8*N = 0.8*2 = 1.6，这个就是原文中说的，同时除以一个值，得到一个小的高斯分布，这就是处理后新的高斯分布的缩放因子
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        # 复制选中的点的旋转信息为 N 组。
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        # 将选中的点的特征和透明度信息分别复制为 N 组。
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        # 调用 densification_postfix 方法，对新生成的点云数据进行后处理。
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        # 创建一个掩码，用于筛选需要保留的点云数据。
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        # 将新生成的高斯分布添加到总的高斯分布中去
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        # selected_pts_mask.sum() 返回布尔掩码 selected_pts_mask 中值为 True 的元素的数量。
        # selected_pts_mask 这是原始的高斯分布的 mask，N * selected_pts_mask.sum() 这是新创建的高斯分布的 mask
        # 构建这个 mask ，剔除掉原始高斯分布中的不符合要求的高斯分布，
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))

        # 剔除操作， 因为这是分裂， 所以要把原先的高斯球删掉
        self.prune_points(prune_filter)

    # 用于密集化和克隆的操作。
    # densify_and_clone 输入为：
    #   第一个参数：grads 当前计算的平均梯度
    #   第二个参数：grad_threshold=0.0002 ，表示 最大梯度阈值
    #   第三个参数：scene_extent，表示场景空间的对角线长度
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        # 用于根据梯度的范数是否大于等于梯度阈值 grad_threshold 来创建一个掩码（mask），表示哪些点满足了密集化的条件。
        # torch.norm(grads, dim=-1): 这部分计算了梯度张量 grads 沿着最后一个维度的范数，即计算了每个梯度向量的大小。
        # torch.norm(grads, dim=-1) >= grad_threshold: 这部分产生了一个布尔型张量，其中每个元素都是对应梯度向量的大小是否大于等于阈值 grad_threshold 的结果。
        # torch.where(condition, x, y): 这是 torch.where 函数的调用，它根据 condition 来选择 x 或 y 的元素。在这里，condition 是上一步产生的布尔型张量，
        # True 代表大于等于阈值，所以对应的元素应该是 True，而 x 是 True，y 是 False，所以如果条件为真，则选取 x，否则选取 y。
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # 这行代码是在上一行代码中创建的选定点掩码 selected_pts_mask 的基础上，再次进行筛选，确保所选点的缩放程度不会过大，以防止点过于密集。
        # torch.max(self.get_scaling, dim=1).values: 这部分代码计算了点云数据中每个点的缩放程度的最大值。self.get_scaling 应该是一个张量，
        # 其中每一行代表一个点，每一列代表一个维度（可能是三维空间中的 x、y、z），而 dim=1 表示沿着每行计算最大值。
        # torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent: 这部分将每个点的最大缩放值与
        # 阈值 self.percent_dense * scene_extent 进行比较，self.percent_dense 是一个百分比值，表示密集度的阈值，而 scene_extent 则是场景的范围。
        # 这个比较操作会产生一个布尔型张量，其中 True 表示对应点的最大缩放值在允许范围内，而 False 表示超出了范围。
        # torch.logical_and(selected_pts_mask, ...: 最后，通过逻辑与操作，将之前的选定点掩码 selected_pts_mask 与新的缩放程度限制条件进行结合，
        # 产生一个新的掩码，其中仅包含满足两个条件的点。
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        # selected_pts_mask 中 掩码 既满足 平均梯度超过设定阈值 又满足 当前高斯最大尺度小于设定阈值。

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    # 剔除
    # 这段代码看起来是一个方法，它的功能包括密集化（densify）、修剪（prune）和克隆（clone）。
    # 输入为：
    #   第一个参数：max_grad=0.0002，表示 最大梯度阈值
    #   第二个参数：min_opacity=0.005，表示最小透明度阈值
    #   第三个参数：extent 表示场景空间的对角线长度
    #   第四个参数：max_screen_size 尺寸阈值，上一行代码定义
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        # grads = self.xyz_gradient_accum / self.denom: 首先，计算了梯度的密度。
        # self.xyz_gradient_accum 是梯度的累积值，而 self.denom 是梯度更新的次数。
        # 这里通过除法计算了梯度的平均密度。若分母为0，产生了NaN，则通过 grads.isnan() 将其置为0.0。
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # 这两行代码调用了两个函数，用于根据梯度密度进行克隆和分割操作。
        # densify_and_clone 输入为：
        #   第一个参数：grads 当前计算的平均梯度
        #   第二个参数：max_grad=0.0002 ，表示 最大梯度阈值
        #   第三个参数：extent，表示场景空间的对角线长度
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # 这里根据不透明度（opacity）小于阈值 min_opacity 的条件创建了一个修剪的掩码（mask）。
        # squeeze() 函数用于移除维度中的大小为1的维度。
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        #  这里检查是否有 max_screen_size 的限制。如果设置了 max_screen_size，
        #  则根据屏幕上的点的大小和世界空间中点的大小进行进一步的修剪。
        if max_screen_size:
            # 高斯球投影二维平面 > 最大屏幕尺寸
            big_points_vs = self.max_radii2D > max_screen_size
            # 高斯求尺寸 >  0.1 * extent 是场景的范围
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # 最后，根据之前创建的修剪掩码 prune_mask，对点进行修剪操作。
        self.prune_points(prune_mask)

        # 清空GPU缓存，释放不再需要的内存。
        # torch.cuda.empty_cache() 是一个 PyTorch 提供的函数，用于清空当前 CUDA 设备上的内存缓存。
        # 在深度学习任务中，特别是在使用大型模型和大量数据时，显存管理非常重要。当显存被占用过多时，可能会导致程序崩溃或者性能下降。
        # 因此，通过定期清空显存缓存，可以释放未使用的显存，从而提高系统的稳定性和性能。
        # 使用 torch.cuda.empty_cache() 可以在需要时手动清空显存缓存。这在进行模型训练过程中的某些关键点，如训练批次结束后或者在内存不足时，可能是有用的。
        torch.cuda.empty_cache()

    # 去添加自适应密度控制过程中的一个状态，
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # 这个就是给每个迭代中可见的部分累加梯度的一个操作
        # viewspace_point_tensor.grad[update_filter,:2]：在这里，viewspace_point_tensor 似乎是一个包含梯度的张量，
        # update_filter 可能是一个布尔型张量或者是一个索引列表，用于指示哪些梯度需要被选择。而 :2 则是选择每个梯度的前两个元素。因为是视图平面上的点，是一个二维的高斯分布（椭圆），所以只需要x和y方向
        # torch.norm(..., dim=-1, keepdim=True)：这个函数计算了所选梯度的范数（大小），dim=-1 表示沿着最后一个维度进行操作，keepdim=True 确保结果张量保留与输入张量相同的维度。
        # 最后，计算出的范数被加到了 self.xyz_gradient_accum[update_filter] 中，这个张量很可能包含了累积的梯度。
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)

        # 这个从实践上来说是存储了每个高斯更新梯度累积的次数,但干什么没看懂
        self.denom[update_filter] += 1