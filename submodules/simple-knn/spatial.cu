/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "spatial.h"
#include "simple_knn.h"

torch::Tensor
distCUDA2(const torch::Tensor& points)
{
// 定义输入点的数量
  const int P = points.size(0);

// points.options(): 获取 points 张量的选项（例如设备类型、内存布局）。
// .dtype(torch::kFloat32): 将张量的数据类型显式设置为 32 位浮点数。
// torch::full({P}, 0.0, float_opts): 创建一个形状为 (P,) 的张量 means，所有元素初始化为 0.0。
  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);

// SimpleKNN 是一个类，knn 是其静态方法。从名称来看，它实现了 K 近邻（k-Nearest Neighbors）的某种计算逻辑。
// 调用格式表明 knn 处理的数据是 C 风格的指针（即数组指针）。
// P: 表示点的数量（点云的大小）。 通常是 points 张量的第一个维度大小 points.size(0)。
// (float3*)points.contiguous().data<float>():
// points: PyTorch 的张量，通常形状为 (P, D)，D 是维度（例如 3 表示 3D 坐标）。
// .contiguous(): 确保张量在内存中是连续存储的（C 顺序布局）。如果张量已经是连续的，则此操作不会复制内存。
// .data<float>(): 获取底层的 float* 数据指针，指向张量的数据。 注意：这会跳过 PyTorch 的自动梯度管理，因此需要谨慎操作。
// (float3*): 将指针类型从 float* 转换为 float3*，表明每一组 3 个浮点数表示一个 3D 点。
// means.contiguous().data<float>(): means: PyTorch 的结果张量，形状为 (P,)。
// .contiguous(): 确保张量是连续的，便于直接访问其底层数据。
// .data<float>(): 获取底层的 float* 数据指针，指向 means 的存储区域。knn 函数可能会通过该指针写入计算结果。
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());
// knn 的可能实现
// 根据函数调用的上下文，SimpleKNN::knn 的功能可能如下：
// 遍历输入点集合 points，计算每个点与其最近邻之间的某种关系（例如距离、索引）。
// 将计算结果写入 means 张量（形状为 (P,)）。
// 使用 float3* 表明每个点由 3 个浮点数（例如 x, y, z）组成，表示 3D 坐标。

  return means;
}