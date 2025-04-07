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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # if load_iteration: 判断是否有指定的加载迭代次数。
        # 如果 load_iteration 不为零、空或者 None，则条件成立。
        if load_iteration:
            # 在条件成立的情况下，进一步检查 load_iteration 的值。如果 load_iteration 的值为 -1
            if load_iteration == -1:
                # 则会调用 searchForMaxIteration 函数来搜索指定路径下的最大迭代次数，结果会存储在 self.loaded_iter 中。
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                # 否则，直接将 load_iteration 的值存储在 self.loaded_iter 中。
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # 如果存在一个叫 sparse 的文件就当作 Colmap 数据载入,如果有 transforms_train.json 就当 Blender 数据载入
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # args.train_test_exp 的作用就是：如果其为true，则训练集中为全部数据；其为false，训练集中不包含测试数据
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # 检查 self.loaded_iter 是否为假（即为False或未定义）。 如果self.loaded_iter为假，则执行以下操作
        if not self.loaded_iter:
            # 把数据集文件夹里的 points3d.ply 复制到模型文件夹里,然后开始从 scene_info 里恢复 camera ,把 scene_info 里
            # 所有的 test 和 train camera 放进一个列表,再调用 camera_to_json 把信息存储成一个 json 文件放进模型文件夹
            #  从 scene_info 中获取点云数据文件路径（scene_info.ply_path）并将其复制到模型文件夹中的input.ply。
            # 从 /sparse/0/points3D.ply 生成 /output/input.ply
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                dest_file.write(src_file.read())
            # 创建一个空列表json_cams来存储相机信息。
            json_cams = []
            # 创建一个空列表 camlist 来存储所有测试和训练相机的列表。
            camlist = []
            if scene_info.test_cameras:
                # 如果存在测试相机信息 (scene_info.test_cameras) ，则将其添加到 camlist 中。
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                # 如果存在训练相机信息 (scene_info.train_cameras) ，也将其添加到 camlist 中。
                camlist.extend(scene_info.train_cameras)
            # 遍历camlist中的每个相机帧，将其转换为JSON格式，并添加到json_cams列表中。
            for id, cam in enumerate(camlist):
                # 这里只写进去 id;image_name;width;heigth;position[1,3];rotation[3,3];fy;fx;
                json_cams.append(camera_to_JSON(id, cam))
            # 将json_cams列表写入到模型文件夹中的cameras.json文件中。
            # 生成 cameras.json
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # 随机打乱训练视图与测试视图
        if shuffle:
            # 这段代码用于在训练视图和测试视图中随机打乱顺序。如果shuffle参数为True，
            # 则会对训练视图和测试视图进行随机打乱。默认就是true
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # 读取 scene_info.nerf_normalization 中存储的 对角线长度
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 往 train_cameras 和 test_cameras 里面写视角
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            # 这里是在 self.train_cameras 字典中使用 resolution_scale 作为键来存储训练视角的信息。
            # cameraList_from_camInfos(): 这是一个函数调用，
            # 它可能会根据传入的参数从 scene_info.train_cameras 中获取相机信息，并根据 resolution_scale 进行处理，最终返回一个相机列表。
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

# 这段代码定义了一个 save 方法，用于保存模型状态到磁盘上。
    def save(self, iteration):
        # 这一行代码构建了一个保存点云数据的路径。它使用了 os.path.join 函数将模型路径 self.model_path
        # 与子目录 point_cloud 以及格式化的迭代次数路径组合在一起，用于保存当前迭代的点云数据。
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        #  这一行代码调用了 self.gaussians 的 save_ply 方法，将点云数据保存为 PLY 格式的文件。
        #  self.gaussians 可能是一个对象，包含了点云数据，并提供了保存点云数据的方法。
        # 综上所述，这段代码的作用是将当前模型的点云数据保存到磁盘上，以 PLY 格式的文件形式存储。
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        # 创建字典 exposure_dict：
        # 遍历 self.gaussians.exposure_mapping 中的所有 image_name。对于每个 image_name，调用
        # self.gaussians.get_exposure_from_name(image_name) 获取其对应的曝光值。
        # detach(): 将张量从计算图中分离，避免梯度计算。
        # cpu(): 将张量移至 CPU 内存（如果最初在 GPU 上）。
        # numpy(): 将张量转换为 NumPy 数组。
        # tolist(): 将 NumPy 数组转换为 Python 列表，便于序列化为 JSON。
        # 结果 ： 生成一个字典 exposure_dict，其键是图像名称，值是对应的曝光参数列表。
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }
        # 使用 os.path.join 将模型路径 self.model_path 和文件名 "exposure.json" 组合成完整路径。
        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            # 写入文件：
            # 打开文件，使用 "w" 模式（写入模式），获取文件句柄 f。
            # 使用 json.dump 将字典 exposure_dict 序列化为 JSON 格式并写入文件。
            # indent=2: 设置缩进级别为 2，以使生成的 JSON 文件更具可读性。
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
