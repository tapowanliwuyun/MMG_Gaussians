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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

# cam_info ：是处理好的训练数据
def getNerfppNorm(cam_info):
    # 这段代码定义了一个函数 get_center_and_diag()，用于计算摄像机中心和场景的对角线长度。
    def get_center_and_diag(cam_centers):
        # cam_centers = np.hstack(cam_centers) 将摄像机中心列表 cam_centers 水平堆叠，以便后续计算。
        cam_centers = np.hstack(cam_centers)
        # 计算摄像机中心的平均值，
        # axis=1 表示沿着水平方向计算平均值，keepdims=True 保持结果的维度不变。
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        # 将平均摄像机中心作为场景的中心点。
        center = avg_cam_center
        # 计算每个摄像机中心到场景中心的距离，
        # np.linalg.norm() 函数用于计算向量的范数，axis=0 表示沿着垂直方向计算，
        # keepdims = True 保持结果的维度不变。
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        # 并取其中的最大值作为场景的对角线长度。
        diagonal = np.max(dist)
        # 返回场景的中心点（已扁平化）和对角线长度。
        return center.flatten(), diagonal

    cam_centers = []
    # 这段代码遍历了 cam_info 中的每一个摄像机信息，并计算了每个摄像机的相机中心坐标。
    # 遍历 cam_info 中的每一个摄像机信息，其中 cam 表示当前遍历到的摄像机信息对象。
    for cam in cam_info:
        # 调用了 getWorld2View2 函数，
        # 传入了摄像机的旋转矩阵 cam.R 和平移向量 cam.T，从而计算了从世界坐标系到相机坐标系的变换矩阵 W2C。
        W2C = getWorld2View2(cam.R, cam.T)
        # 使用 np.linalg.inv() 函数计算了 W2C 的逆矩阵，
        # 从而得到了从相机坐标系到世界坐标系的变换矩阵 C2W。
        C2W = np.linalg.inv(W2C)
        # 将变换矩阵 C2W 的前三行中的第四个元素（即相机中心的坐标）添加到 cam_centers 列表中。
        cam_centers.append(C2W[:3, 3:4])

    #  调用了 get_center_and_diag 函数，
    # 传入了摄像机中心坐标列表 cam_centers，并获取了返回的场景中心点坐标 center 和对角线长度 diagonal
    center, diagonal = get_center_and_diag(cam_centers)
    # 计算了新的半径 radius，将对角线长度 diagonal 扩大了 1.1 倍。
    radius = diagonal * 1.1

    # 计算了平移向量 translate，将场景的中心点坐标取反，得到的向量表示了将场景的中心点移动到原点所需要的平移量。
    translate = -center

    return {"translate": translate, "radius": radius}

# cam_extrinsics 读取的外参信息
# cam_intrinsics 读取的内参信息
# images_folder  图像的路径，默认是 colmap 计算后得到的 images 下的路径
def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    #  Python 中的 enumerate() 函数，用于在迭代过程中同时获取索引值和元素值。
    # 其中 idx 是元素在迭代过程中的索引，而 key 则是对应的元素值。
    for idx, key in enumerate(cam_extrinsics):
        # 这段代码使用了 sys.stdout.write() 来实现一种动态更新的输出效果。
        # sys.stdout.write('\r') 使用回车符 \r，将光标移动到当前行的开头，实现覆盖当前行的效果。
        sys.stdout.write('\r')
        # the exact output you're looking for:
        # 然后使用 sys.stdout.write() 输出一个格式化的字符串，其中包含了迭代的进度信息，
        # 格式为 "Reading camera {}/{}"，其中 {} 部分会被 idx+1 和 len(cam_extrinsics) 替换，表示当前迭代的进度。
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        # sys.stdout.flush() 用于强制刷新标准输出缓冲区，确保在输出完信息后立即将其显示出来。
        sys.stdout.flush()

        # extr = cam_extrinsics[key] 获取了字典 cam_extrinsics 中键为 key 的值，
        # 这个值可能是某种摄像机的外参信息。
        extr = cam_extrinsics[key]
        # intr = cam_intrinsics[extr.camera_id] 获取了字典 cam_intrinsics 中键
        # 为 extr.camera_id 的值，
        # 这个值可能是对应摄像机的内参信息。extr.camera_id 可能是外参信息中表示摄像机ID的属性。
        intr = cam_intrinsics[extr.camera_id]
        # height = intr.height 和 width = intr.width 分别获取了内参信息中的摄像机高度和宽度
        height = intr.height
        width = intr.width

        # uid = intr.id 获取了内参信息中的摄像机ID，这可能用于标识不同的摄像机。
        uid = intr.id
        # R = np.transpose(qvec2rotmat(extr.qvec)) 根据外参信息中的四元数 (extr.qvec)，
        # 使用了 qvec2rotmat 函数来计算旋转矩阵 R。
        # qvec2rotmat 函数可能用于将四元数转换为旋转矩阵，并且对结果进行了转置操作。
        R = np.transpose(qvec2rotmat(extr.qvec))
        # T = np.array(extr.tvec) 获取了外参信息中的平移向量 tvec，并将其转换为 NumPy 数组形式。
        T = np.array(extr.tvec)

        #这段代码根据摄像机的模型类型（intr.model）计算了视场角（Field of View，FOV）。
        if intr.model=="SIMPLE_PINHOLE":
            # 如果摄像机模型是 "SIMPLE_PINHOLE"，则假定只有一个焦距参数，
            # 从 intr.params 中获取第一个参数作为 focal_length_x，
            # 然后使用 focal2fov() 函数分别计算水平和垂直方向的 FOV。
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            # 如果摄像机模型是 "PINHOLE"，则假定有两个焦距参数，分别是水平和垂直方向的焦距，
            # 从 intr.params 中获取这两个参数作为 focal_length_x 和 focal_length_y，然后分别计算水平和垂直方向的 FOV。
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            # 如果摄像机模型既不是 "SIMPLE_PINHOLE" 也不是 "PINHOLE"，则断言失败，表示不支持这种摄像机模型。
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        # 这段代码构建了图像的路径，并加载了图像。
        # image_path = os.path.join(images_folder, os.path.basename(extr.name)) 构建了图像的完整路径。
        # images_folder 是图像所在文件夹的路径，如 \1_2_10Hz_hubinlu_\images\
        # extr.name 是外参信息中的图像名称，将其与 images_folder 拼接成完整的图像路径。
        image_path = os.path.join(images_folder, extr.name)
        # 获取了图像的名称
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        # 这段代码创建了一个名为 cam_info 的 CameraInfo 对象，并将其添加到了一个名为 cam_infos 的列表中。
        # 创建了一个 CameraInfo 对象，传入了摄像机的各种信息，如摄像机ID（uid）、
        # 旋转矩阵（R）、平移向量（T）、水平和垂直方向的视场角（FovY 和 FovX）、深度参数（depth_params）、
        # 图像对象（image）、图像路径（image_path）、图像名称（image_name）、深度路径（depth_path）、
        # 图像宽度（width）和高度（height）和该图片是否为测试数据集数据布尔量（is_test）等。
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        # 将创建的 CameraInfo 对象添加到了名为 cam_infos 的列表中，
        # 这样列表中就包含了多个摄像机的信息，可以在后续的代码中进行进一步处理或者分析。
        cam_infos.append(cam_info)
    # 输出一个换行符，使得接下来的输出从新的一行开始。
    sys.stdout.write('\n')
    return cam_infos

# 这个函数 fetchPly 的作用是从指定路径加载一个 .ply 文件，
# 并将其中的顶点、颜色和法线信息提取出来，然后返回一个基本的点云对象。
def fetchPly(path):
    # 使用 PlyData 类的 read 方法读取指定路径的 .ply 文件，得到一个 PlyData 对象 plydata，
    # 其中包含了文件中的所有数据。
    plydata = PlyData.read(path)
    # 从 PlyData 对象中提取了名为 'vertex' 的元素，该元素包含了文件中的顶点信息。
    vertices = plydata['vertex']
    # 从顶点元素中提取了 x、y、z 坐标信息，
    # 并使用 np.vstack() 函数将它们堆叠成一个二维数组，然后通过转置 .T 得到了点的位置信息。
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    # 从顶点元素中提取了红、绿、蓝（RGB）颜色信息，
    # 并将其归一化到 [0, 1] 范围内，然后通过类似的方法得到了颜色信息。
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # 从顶点元素中提取了法线向量的 x、y、z 分量，
    # 并将其堆叠成一个二维数组，得到了法线向量的信息。
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T

    # 最后，根据提取出来的位置、颜色和法线信息，创建了一个 BasicPointCloud 对象，其中包含了点云的位置、颜色和法线信息。
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

# 这个函数 storePly 用于将点云数据存储为 .ply 格式的文件。
def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    # 首先定义了一个结构化数组的数据类型 dtype，包含了点的 XYZ 坐标和法线向量的 XYZ 分量，以及颜色的 RGB 分量。
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    # 然后创建了一个与输入点云数据 xyz 相同大小的零矩阵 normals，用于存储法线向量。这里假设法线向量暂时为空。
    normals = np.zeros_like(xyz)

    # 创建了一个空数组 elements，用于存储结构化数组的数据。结构化数组中包含了点的 XYZ 坐标、法线向量和颜色信息。
    # elements，其大小与输入的点云数据 xyz 的行数相同，数据类型由之前定义的 dtype 确定。
    elements = np.empty(xyz.shape[0], dtype=dtype)
    # 将点的 XYZ 坐标、法线向量和 RGB 颜色信息按列连接成一个完整的属性数组 attributes。
    # 这里假设法线向量为空，因此使用了之前创建的 normals 数组，其大小与 xyz 相同。
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    # 将属性数组 attributes 中的每一行转换为元组，
    # 并将转换后的元组赋值给结构化数组 elements。这里使用了 map 函数来对属性数组的每一行进行转换
    # ，并将结果转换为列表，然后赋值给结构化数组 elements。
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    # 这段代码创建了一个 PlyData 对象，并将点云数据写入到 .ply 文件中。
    #  使用 PlyElement.describe() 函数创建了一个描述点云数据的 PlyElement 对象 vertex_element。
    # 这个对象描述了点云数据中的顶点信息，并使用了之前创建的结构化数组 elements。
    vertex_element = PlyElement.describe(elements, 'vertex')
    # 创建了一个 PlyData 对象 ply_data，
    # 其中包含了描述点云数据的 vertex_element。这样就创建了一个包含点云数据的 PlyData 对象。
    ply_data = PlyData([vertex_element])
    # 将创建的 PlyData 对象写入到指定路径 path 对应的 .ply 文件中。
    # 这样就将点云数据保存到了 .ply 文件中。
    ply_data.write(path)

# path : source 路径
# images ： 图像文件夹名称“images”
# eval ：是否评价
# train_test_exp ：
# llffhold ： 训练评价分比
def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    # 从images.bin中读出相机外参,从cameras.bin中读出相机内参
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    # 如果需要训练,则每隔 llffhold=8 张抽一张视图做测试视图
    if eval:
        # 如果 path 中包含字符串 "360"，则设置 llffhold 的值为 8。
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            # 从 cam_extrinsics 中提取每个相机的名称。
            # cam_extrinsics 应该是一个字典或类似结构，键为 cam_id，值包含相机的外部参数。
            # 通过 cam_extrinsics[cam_id].name 提取每个相机的名称。
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            # 对提取的相机名称进行排序。
            cam_names = sorted(cam_names)
            # 选择排序后的相机名称，步长为 llffhold。
            # 这意味着从相机名称中每隔 llffhold 个选择一个，生成测试用的相机名称列表。
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            # 打开 path/sparse/0/test.txt 文件，读取其中的内容。path 是给定的路径。
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                # 从文件 test.txt 中读取每一行，并去掉行尾的空白字符（如换行符），
                # 将这些行组成 test_cam_names_list。
                test_cam_names_list = [line.strip() for line in file]
    else:
        # 如果 eval 为假（False），则直接将 test_cam_names_list 设置为空列表。
        test_cam_names_list = []
    reading_dir = "images" if images == None else images
    # 调用 readColmapCameras 里读出还未排序的相机信息,再按图片名称对 camera 排序
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir),
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    # 这行代码对 cam_infos_unsorted 中的摄像机信息列表进行了排序，并将结果存储在 cam_infos 中。
    # 排序是根据摄像机信息中的图像名称来进行的。
    # sorted() 函数用于对可迭代对象进行排序，它接受一个可迭代对象作为参数，并返回一个新的已排序的列表。
    # cam_infos_unsorted.copy() 通过调用 copy() 方法创建了 cam_infos_unsorted 列表的一个副本，以免直接对原列表进行修改。
    # key=lambda x: x.image_name 指定了排序的关键字，即按照摄像机信息中的 image_name 属性进行排序。
    # 这里使用了一个匿名函数 lambda，它接受一个参数 x，表示列表中的每个元素，然后返回 x.image_name 作为排序的依据。
    # 最终，cam_infos 变量存储了按照图像名称排序后的摄像机信息列表。
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # 这两行代码是根据条件将 cam_infos 中的相机信息分为训练集和测试集
    # train_test_exp 为 True ： 忽略 c.is_test 的值，将所有相机 (cam_infos) 都包含到 train_cam_infos 中。
    # train_test_exp 为 False ： 仅选择 c.is_test 为 False 的相机，表示只选择非测试集的相机。
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    # 仅选择 c.is_test 为 True 的相机信息。
    test_cam_infos = [c for c in cam_infos if c.is_test]

    # 调用 getNerfppNorm 求一个 nerf_normalization ,但还是不知道什么意思
    # train_cam_infos 根据其中的相机位姿信息，计算 nerf_normalization  {-场景中心点坐标，对角线长度}
    nerf_normalization = getNerfppNorm(train_cam_infos)


    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    # 从bin或者txt转化出需要的points3D.ply文件,用处不是特别大
    # 这段代码检查是否存在指定路径的 .ply 文件。如果不存在，它会尝试从 .bin 文件或 .txt 文件读取点云数据，然后将数据存储为 .ply 文件。
    # 通过 os.path.exists() 函数检查指定路径的 .ply 文件是否存在，如果不存在则执行下面的操作。
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        # 在 try 语句中，尝试使用 read_points3D_binary() 函数从 .bin 文件中读取点云数据。如果读取失败，会抛出异常。
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            # 如果 read_points3D_binary() 抛出异常（无法从 .bin 文件中读取数据），
            # 则会执行 except 语句块中的代码，尝试从 .txt 文件中读取点云数据。
            xyz, rgb, _ = read_points3D_text(txt_path)
        # 将读取到的点云数据存储为 .ply 文件。这个函数可能会使用点云的 XYZ 坐标和 RGB 颜色信息来创建 .ply 文件。
        storePly(ply_path, xyz, rgb)
    try:
        # 这个函数的作用是加载一个 .ply 文件，并返回一个包含了点云信息的基本点云对象。
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # 通过上述数据初始化一个SceneInfo
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}