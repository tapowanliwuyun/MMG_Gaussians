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

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    # 遍历文件夹中的所有文件名，并使用列表推导式从每个文件名中提取迭代次数。
    # 假设文件名遵循特定格式，其中包含迭代次数的信息。
    # 这里假设文件名的格式是以某个前缀开头，后跟下划线和迭代次数，例如“prefix_iterX”，其中“X”表示迭代次数。
    # 将提取的迭代次数转换为整数，并将它们存储在列表 saved_iters 中。
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    # 使用内置的 max() 函数找到列表 saved_iters 中的最大值，即找到已保存模型的最大迭代次数。
    # 将最大迭代次数作为结果返回。
    return max(saved_iters)
