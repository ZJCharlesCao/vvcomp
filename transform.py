from collections import namedtuple
import numpy as np
from plyfile import PlyData, PlyElement
import copy
import hashlib
from collections import defaultdict
from seg import compute_point_cloud_bounds, PointKDTree, get_cb
import torch
import os
from glob import glob
from tqdm import tqdm


# def build_video(axis, pixel_idx_pairs, attributes, ranges):
#     """
#     改进的构建视频张量函数
#
#     参数:
#     axis: int - 处理的轴 (0, 1, 或 2)
#     pixel_idx_pairs: list of tuples - 像素坐标和索引对的列表
#     attributes: tensor - 属性张量
#     ranges: list of tuples - 每个维度的范围
#
#     返回:
#     torch.Tensor - 构建的视频张量
#     """
#     # 确定输出张量的形状
#     tensor_shapes = {
#         0: (round(48 / 3), ranges[0][1], 3, ranges[1][1], ranges[2][1]),
#         1: (round(48 / 3), ranges[1][1], 3, ranges[0][1], ranges[2][1]),
#         2: (round(48 / 3), ranges[2][1], 3, ranges[0][1], ranges[1][1])
#     }
#
#     # 初始化输出张量
#     output_tensor = torch.zeros(tensor_shapes[axis], dtype=torch.float32)
#
#     # 创建映射以优化访问
#     pixel_map = defaultdict(list)
#     for p, idx in pixel_idx_pairs:
#         if axis == 0:
#             pixel_map[p[0]].append((p[1], p[2], idx))
#         elif axis == 1:
#             pixel_map[p[1]].append((p[0], p[2], idx))
#         else:  # axis == 2
#             pixel_map[p[2]].append((p[0], p[1], idx))
#
#     # 填充张量
#     time_steps = round(48 / 3)
#     for key, pixels in pixel_map.items():
#         for x, y, idx in pixels:
#             attribute = attributes[idx]
#             for t in range(time_steps):
#                 color_values = attribute[t * 3 + 3:t * 3 + 6]
#                 if axis == 0:
#                     output_tensor[t, key, :, x, y] = color_values
#                 elif axis == 1:
#                     output_tensor[t, key, :, x, y] = color_values
#                 else:  # axis == 2
#                     output_tensor[t, key, :, x, y] = color_values
#
#     return output_tensor


def buildVideoTensor(axis, points, attributes, ranges):
    """
    改进的构建视频张量函数

    参数:
    axis: int - 处理的轴 (0, 1, 或 2)
    pixel_idx_pairs: list of tuples - 像素坐标和索引对的列表
    attributes: tensor - 属性张量
    ranges: list of tuples - 每个维度的范围

    返回:
    torch.Tensor - 构建的视频张量
    """
    #quantize points
    col_attr = attributes[:, :48]
    geo_attr = attributes[:, 48:56]

    # 确定输出张量的形状
    point_range = {
        0: [[0, ranges[0][1]], [0, ranges[1][1]], [0, ranges[2][1]]],
        1: [[0, ranges[1][1]], [0, ranges[0][1]], [0, ranges[2][1]]],
        2: [[0, ranges[2][1]], [0, ranges[0][1]], [0, ranges[1][1]]]
    }
    num_col = round(col_attr.shape[1] / 3)
    col_tensor_shapes = {
        0: (num_col, ranges[0][1], 3, ranges[1][1], ranges[2][1]),
        1: (num_col, ranges[1][1], 3, ranges[0][1], ranges[2][1]),
        2: (num_col, ranges[2][1], 3, ranges[0][1], ranges[1][1])
    }
    num_geo = round(geo_attr.shape[1] / 3)
    geo_tensor_shapes = {
        0: (num_geo, ranges[0][1], 3, ranges[1][1], ranges[2][1]),
        1: (num_geo, ranges[1][1], 3, ranges[0][1], ranges[2][1]),
        2: (num_geo, ranges[2][1], 3, ranges[0][1], ranges[1][1])
    }

    # 初始化输出张量
    col_tensor = torch.zeros(col_tensor_shapes[axis])
    geo_tensor = torch.zeros(geo_tensor_shapes[axis])

    space_range = compute_point_cloud_bounds(points)
    print(space_range)
    tree = PointKDTree(points, space_range, point_range[axis])
    pixel_idx_pairs= get_cb(tree.root)


    # 创建映射以优化访问
    pixel_map = defaultdict(list)
    for p, idx in pixel_idx_pairs:
        if axis == 0:
            pixel_map[p[0]].append((p[1], p[2], idx))
        elif axis == 1:
            pixel_map[p[1]].append((p[0], p[2], idx))
        else:  # axis == 2
            pixel_map[p[2]].append((p[0], p[1], idx))

    # 填充张量

    for key, pixels in pixel_map.items():
        for x, y, idx in pixels:
            attribute = attributes[idx]
            for t in range(num_col):
                color_values = attribute[t * 3 :t * 3 + 3]
                col_tensor[t, key, :, x, y] = color_values
                if t < num_geo :
                    if t*3+51 > len(attribute):
                        geo_values = torch.cat((attribute[t * 3 + 48:],torch.zeros(3-len(attribute[t * 3 + 48:]))))
                    else:
                        geo_values = attribute[t * 3 + 48:t * 3 + 51]
                    geo_tensor[t, key, :, x, y] = geo_values
    return col_tensor, geo_tensor, pixel_idx_pairs


def extract_attributes(
        tensor_c: torch.Tensor,
        tnesor_g: torch.Tensor,
        pixel_idx_pairs,
        axis: int,
        old_attr: torch.Tensor
) -> torch.Tensor:
    """
    使用向量化操作从张量中提取新的属性

    参数:
    tensor: shape (time_steps, dim1, 3, dim2, dim3) 的张量
    pixel_idx_pairs: 像素坐标和索引对的列表 [(pixel_coords, idx), ...]
    axis: 处理的轴 (0, 1, 或 2)
    old_att: 原始属性张量，shape

    返回:
    torch.Tensor: 更新后的属性张量
    """
    time_steps_c, *dims = tensor_c.shape
    time_steps_g, *dims = tnesor_g.shape
    num_pixels = len(pixel_idx_pairs)

    # 将pixel_idx_pairs转换为张量形式以便批处理
    pixels = torch.tensor([p[0] for p in pixel_idx_pairs])
    indices = torch.tensor([p[1] for p in pixel_idx_pairs])

    # 创建输出张量，复制原始属性
    output_attributes = old_attr.clone()
    output_attributes = torch.cat((output_attributes, torch.zeros((output_attributes.shape[0], 1))),dim=1)
    # output_attributes = torch.zeros((num_pixels, 6 + time_steps * 3), dtype=torch.float32)
    # 根据不同的轴创建索引张量
    if axis == 0:
        batch_indices = pixels[:, 0]
        height_indices = pixels[:, 1]
        width_indices = pixels[:, 2]
    elif axis == 1:
        batch_indices = pixels[:, 1]
        height_indices = pixels[:, 0]
        width_indices = pixels[:, 2]
    else:  # axis == 2
        batch_indices = pixels[:, 2]
        height_indices = pixels[:, 0]
        width_indices = pixels[:, 1]

    # 为每个时间步骤提取颜色值
    for t in tqdm(range(time_steps_c)):
        # 使用高级索引一次性提取所有像素的颜色值
        color_values = tensor_c[t, batch_indices, :, height_indices, width_indices]

        # 更新输出属性张量
        output_attributes[indices, 3 + t * 3:3 + (t + 1) * 3] = torch.tensor(color_values.astype(np.int32),dtype=torch.float32)

        if t < time_steps_g:
            # 使用高级索引一次性提取所有像素的颜色值
            geo_values = tnesor_g[t, batch_indices, :, height_indices, width_indices]
            # 更新输出属性张量
            output_attributes[indices, 51 + t * 3:51 + (t + 1) * 3] = torch.tensor(geo_values.astype(np.int32),dtype=torch.float32)

    return output_attributes[:, :59]

