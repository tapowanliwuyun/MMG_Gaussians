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
import logging
from argparse import ArgumentParser
import shutil

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
#parser.add_argument() 是用于向脚本的参数解析器添加新的命令行参数的函数。
#--no_gpu 是参数的名称，在命令行中使用时需要指定。
# action='store_true' 表示当命令行中包含 --no_gpu 参数时，其值将被设置为 True。
# 如果未指定 --no_gpu 参数，则其值默认为 False。
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
#这行代码创建了一个 colmap_command 变量，用于存储 COLMAP 命令的执行路径。它的工作原理如下：
#如果 args.colmap_executable 的长度大于0（即用户提供了 COLMAP 可执行文件的路径），则使用 args.colmap_executable 的值作为 COLMAP 命令的路径。
#如果 args.colmap_executable 的长度等于0（即用户没有提供 COLMAP 可执行文件的路径），则使用默认的 "colmap" 命令。
#通过这个逻辑，代码可以适应用户提供或不提供 COLMAP 可执行文件路径的情况。
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
# 经过简单的测试，使用gpu比不使用gpu能快10倍
use_gpu = 1 if not args.no_gpu else 0
#它检查是否需要跳过匹配步骤
if not args.skip_matching:
    # 如果不需要跳过，就会创建一个名为 distorted/sparse 的文件夹，用于存储处理后的图像
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    ## 这段代码是用于执行特征提取的部分。
    # 它构建了一个命令字符串 feat_extracton_cmd，然后使用 os.system 函数执行该命令。

    # colmap_command: 这是之前定义的 COLMAP 命令的路径。
    # feature_extractor ": 这部分是 COLMAP 中的特征提取器命令。
    # --database_path: 指定数据库路径，这是 COLMAP 用来存储图像和特征信息的地方。
    #   args.source_path + "/distorted/database.db": 这是数据库路径，由输入参数 args.source_path 和固定的 /distorted/database.db 组成。
    # --image_path: 指定图像路径，即包含待处理图像的文件夹路径。 args.source_path + "/input": 这是图像路径，由输入参数 args.source_path 和固定的 /input 组成。
    # --ImageReader.single_camera 1: 这是一个 COLMAP 参数，用于告诉 COLMAP 所有输入图像都来自同一台相机。
    # --ImageReader.camera_model: 这是指定相机模型的参数，它的值是从输入参数 args.camera 中获取的。  args.camera: 这是用户提供的相机模型。
    # --SiftExtraction.use_gpu: 这是一个 COLMAP 参数，用于指定是否使用 GPU 进行 SIFT 特征提取。
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    # os.system() 函数用于在操作系统上执行命令，并返回命令执行后的退出状态码。
    # 执行 os.system() 后，脚本会等待命令执行完成。完成后，exit_code 变量将存储命令的退出状态码，
    # 通常情况下，0 表示命令成功执行，非零值表示命令执行失败。
    # 这里执行的是特征提取操作，会生成/distorted/database.db，图像和特征信息都存储在里面
    exit_code = os.system(feat_extracton_cmd)

    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    #这段代码是用于执行特征匹配的部分。它构建了一个命令字符串 feat_matching_cmd，然后通过操作系统执行这个命令。
    # colmap_command: 这是之前定义的 COLMAP 命令的路径。
    # exhaustive_matcher ": 这部分是 COLMAP 中的特征匹配器命令。
    #--database_path: 指定数据库路径，这是 COLMAP 用来存储图像和特征信息的地方。
    # args.source_path + "/distorted/database.db": 这是数据库路径，由输入参数 args.source_path 和固定的 /distorted/database.db 组成。
    #--SiftMatching.use_gpu: 这是一个 COLMAP 参数，用于指定是否使用 GPU 进行 SIFT 特征匹配。
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    # 没有多余的文件，所以猜测匹配信息应该是存储在 /distorted/database.db 中了
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    #这段代码是用于执行捆绑调整（Bundle Adjustment）的部分。
    # 它构建了一个命令字符串 mapper_cmd，然后通过操作系统执行这个命令。
    #colmap_command: 这是之前定义的 COLMAP 命令的路径。 " mapper ": 这部分是 COLMAP 中的捆BA命令。
    # --database_path: 指定数据库路径，这是 COLMAP 用来存储图像和特征信息的地方。
    # args.source_path + "/distorted/database.db": 这是数据库路径，由输入参数 args.source_path 和固定的 /distorted/database.db 组成。
    # --image_path: 指定图像路径，即包含待处理图像的文件夹路径。
    # args.source_path + "/input": 这是图像路径，由输入参数 args.source_path 和固定的 /input 组成。
    # --output_path: 指定输出路径，即捆绑调整后的稀疏点云和相机位姿的存储位置。
    # args.source_path + "/distorted/sparse": 这是输出路径，由输入参数 args.source_path 和固定的 /distorted/sparse 组成。
    #--Mapper.ba_global_function_tolerance=0.000001: 这是一个 COLMAP 参数，用于指定捆绑调整中的全局函数容差。通过减小这个容差，可以加快捆绑调整的速度。
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    # 在 /distorted/sparse/0 下 生成4个文件 cameras.bin、images.bin、points3D.bin 和 project.ini
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
# 这段代码是用于图像去畸变（Image undistortion）的部分。
# 它构建了一个命令字符串 img_undist_cmd，然后通过操作系统执行这个命令。
#colmap_command: 这是之前定义的 COLMAP 命令的路径。
# " image_undistorter ": 这部分是 COLMAP 中的图像去畸变器命令。
# --image_path: 指定输入图像路径，即包含待处理图像的文件夹路径。
# args.source_path + "/input": 这是输入图像路径，由输入参数 args.source_path 和固定的 /input 组成。
# --input_path: 指定输入路径，即需要去畸变的图像和相机姿态的保存路径。
# args.source_path + "/distorted/sparse/0": 这是输入路径，由输入参数 args.source_path、固定的 /distorted/sparse/ 和相机索引 0 组成。这里假设相机索引为 0。
# --output_path: 指定输出路径，即去畸变后的图像的保存路径。 args.source_path: 这是输出路径，由输入参数 args.source_path 组成。
# --output_type COLMAP: 这是一个 COLMAP 参数，用于指定输出图像的类型为 COLMAP 格式，这里好像是jpg。
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
# 在input同级别目录，生成文件 run-colmap-geometric.sh 和 run-colmap-photometric.sh，
# 以及文件夹 images、sparse 和 stereo
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse") # 0/ 、cameras.bin、 images.bin 和 points3D.bin
#这行代码是用于创建目录的，它将创建一个路径为 args.source_path + "/sparse/0" 的目录。
# 如果该目录已经存在，将不会抛出错误，而是继续执行。
# exist_ok这个参数告诉 os.makedirs() 函数如果目录已经存在，不要抛出错误。
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
# 这个循环就是将/sparse/下的文件移动到/sparse/0/下
for file in files:
    # 如果文件名为 '0'，则跳过继续处理下一个文件。
    if file == '0':
        continue
    # 这行代码使用了 Python 的 os.path.join() 函数来构建源文件的路径。
    # args.source_path: 这是输入参数，表示源文件的根路径。
    # "sparse": 这是固定的文件夹名，表示目标文件位于源文件根路径下的 sparse 文件夹中。
    # file: 这是当前文件的文件名。
    # 通过 os.path.join() 函数，将这三个部分拼接在一起，生成了源文件的完整路径。
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    # 使用 shutil.move() 函数将源文件移动到目标文件路径。这会将文件从 source_file 移动到 destination_file。
    shutil.move(source_file, destination_file)

if(args.resize):
    # 这是一个条件语句，检查用户是否选择了缩放选项。
    print("Copying and resizing...")

    # Resize images.
    # 这些行代码用于创建三个目标文件夹，分别用于存储缩放后不同比例的图像。
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    # 这行代码获取了源文件夹中所有图像文件的列表。
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")
