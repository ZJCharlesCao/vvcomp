# from collections import namedtuple
# from distutils.dep_util import newer
#
# import numpy as np
# from networkx import maximum_flow_value
# from numpy.distutils.lib2def import output_def
# from plyfile import PlyData, PlyElement
# import copy
# import hashlib
# from collections import defaultdict
# import pandas as pd
# from scipy.cluster.hierarchy import single
# from scipy.sparse import coo_matrix
#
# from seg import compute_point_cloud_bounds, PointKDTree, get_cb
# import torch
# import os
# from glob import glob
# from tqdm import tqdm
#
#
# def log_transform(coords):
#     positive = coords > 0
#     negative = coords < 0
#     zero = coords == 0
#
#     transformed_coords = np.zeros_like(coords)
#     transformed_coords[positive] = np.log1p(coords[positive])
#     transformed_coords[negative] = -np.log1p(-coords[negative])
#     # For zero, no change is needed as transformed_coords is already initialized to zeros
#
#     return transformed_coords
#
# def inverse_log_transform(transformed_coords):
#     positive = transformed_coords > 0
#     negative = transformed_coords < 0
#     zero = transformed_coords == 0
#
#     original_coords = np.zeros_like(transformed_coords)
#     original_coords[positive] = np.expm1(transformed_coords[positive])
#     original_coords[negative] = -np.expm1(-transformed_coords[negative])
#     # For zero, no change is needed as original_coords is already initialized to zeros
#
#     return original_coords
#
#
# def quantization(attributes, enable = True):
#     def single_q(attr,info_table):
#         min_values, max_values = torch.min(attr).numpy(), torch.max(attr).numpy()
#         q_attr = torch.clamp(((attr - min_values) * 65535 / (max_values - min_values)), 0, 65535).to(torch.int32)
#         info_table.loc[len(info_table)] = min_values,max_values
#         return q_attr, info_table
#
#     info = pd.DataFrame(columns=['Column1', 'Column2'])
#     ex_attr_xyz = attributes[:, :3]
#     ex_attr_f = attributes[:, 3:6]
#     ex_attr_rest = attributes[:, 6:6 + 3 * (3 + 1) ** 2 - 3]
#     ex_attr_opa, ex_attr_scale, ex_attr_rot = (attributes[:,6 + 3 * (3 + 1) ** 2 - 3:6 + 3 * (3 + 1) ** 2 - 2],
#                                                attributes[:,6 + 3 * (3 + 1) ** 2 - 2:6 + 3 * (3 + 1) ** 2 + 1],
#                                                attributes[:, 6 + 3 * (3 + 1) ** 2 + 1:6 + 3 * (3 + 1) ** 2 + 5])
#
#     t_attr_xyz = torch.Tensor(log_transform(ex_attr_xyz))
#     min_values,max_values = torch.min(t_attr_xyz),torch.max(t_attr_xyz)
#     info.loc[len(info)] = min_values,max_values
#     q_attr_xyz = torch.clamp(((t_attr_xyz - min_values) * 65535 / (max_values - min_values)), 0, 65535).to(torch.int32)
#
#     q_attr_fdc, info = single_q(ex_attr_f, info)
#     q_attr_rest, info = single_q(ex_attr_rest, info)
#     q_attr_opa, info = single_q(ex_attr_opa, info)
#     q_attr_scale, info = single_q(ex_attr_scale, info)
#     q_attr_rot, info = single_q(ex_attr_rot, info)
#
#     q_attr_col = torch.cat([q_attr_fdc, q_attr_rest], dim=1)
#     q_attr_geo = torch.cat([q_attr_opa, q_attr_scale, q_attr_rot],dim=1)
#
#     return q_attr_xyz, q_attr_col, q_attr_geo, info
#
# def inverse_quantization(q_attributes, info):
#
#     min_values, max_values = info.loc[0]
#     t_attr_xyz = q_attributes[:, :3] * (max_values - min_values) / 65535 + min_values
#     ex_attr_xyz = inverse_log_transform(t_attr_xyz)
#     ex_attr_xyz = torch.from_numpy(ex_attr_xyz)
#     min_values, max_values = info.loc[1]
#     ex_attr_f = q_attributes[:, 3:6] * (max_values - min_values) / 65535 + min_values
#     min_values, max_values = info.loc[2]
#     ex_attr_rest = q_attributes[:, 6:6 + 3 * (3 + 1) ** 2 - 3] * (max_values - min_values) / 65535 + min_values
#     min_values, max_values = info.loc[3]
#     ex_attr_opa = q_attributes[:, 6 + 3 * (3 + 1) ** 2 - 3] * (max_values - min_values) / 65535 + min_values
#     ex_attr_opa = ex_attr_opa.unsqueeze(1)
#     min_values, max_values = info.loc[4]
#     ex_attr_scale = q_attributes[:, 6 + 3 * (3 + 1) ** 2 - 2:6 + 3 * (3 + 1) ** 2 + 1] * (max_values - min_values) / 65535 + min_values
#     min_values, max_values = info.loc[5]
#     ex_attr_rot = q_attributes[:, 6 + 3 * (3 + 1) ** 2 + 1:6 + 3 * (3 + 1) ** 2 + 5] * (max_values - min_values) / 65535 + min_values
#     new_attr = torch.cat([ex_attr_xyz, ex_attr_f, ex_attr_rest, ex_attr_opa, ex_attr_scale, ex_attr_rot], dim=1)
#     return new_attr
#
#
# def buildVideoTensor(axis, attributes, ranges):
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
#     #quantize points
#     points = attributes[:,0:3]
#     coord_attr,col_attr,geo_attr,para_info =quantization(attributes)
#
#     # 确定输出张量的形状
#     point_range = {
#         0: [[0, ranges[0][1]], [0, ranges[1][1]], [0, ranges[2][1]]],
#         1: [[0, ranges[1][1]], [0, ranges[0][1]], [0, ranges[2][1]]],
#         2: [[0, ranges[2][1]], [0, ranges[0][1]], [0, ranges[1][1]]]
#     }
#     num_col = round(col_attr.shape[1] / 3)
#     col_tensor_shapes = {
#         0: (num_col, ranges[0][1], 3, ranges[1][1], ranges[2][1]),
#         1: (num_col, ranges[1][1], 3, ranges[0][1], ranges[2][1]),
#         2: (num_col, ranges[2][1], 3, ranges[0][1], ranges[1][1])
#     }
#     num_geo = round(geo_attr.shape[1] / 3)
#     geo_tensor_shapes = {
#         0: (num_geo, ranges[0][1], 3, ranges[1][1], ranges[2][1]),
#         1: (num_geo, ranges[1][1], 3, ranges[0][1], ranges[2][1]),
#         2: (num_geo, ranges[2][1], 3, ranges[0][1], ranges[1][1])
#     }
#
#     # 初始化输出张量
#     col_tensor = torch.zeros(col_tensor_shapes[axis])
#     geo_tensor = torch.zeros(geo_tensor_shapes[axis])
#     coord_tensor = torch.zeros((1, ranges[0][1], 3, ranges[1][1], ranges[2][1]), dtype=torch.float32)
#
#     space_range = compute_point_cloud_bounds(points)
#     print(space_range)
#     tree = PointKDTree(points, space_range, point_range[axis])
#     pixel_idx_pairs= get_cb(tree.root)
#
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
#
#     for key, pixels in pixel_map.items():
#         for x, y, idx in pixels:
#             coord_tensor[0, key, :, x, y] = torch.tensor(coord_attr[idx],dtype=torch.float32)
#             col_attr_p = col_attr[idx]
#             geo_attr_p = geo_attr[idx]
#             for t in range(num_col):
#                 color_values = col_attr_p[t * 3 :t * 3 + 3]
#                 col_tensor[t, key, :, x, y] = color_values
#                 if t < num_geo :
#                     if t*3+3 > geo_attr.shape[1]:
#                         geo_values = torch.cat((geo_attr_p[t * 3:],torch.zeros(3 - (geo_attr.shape[1] - t * 3),dtype=torch.float32)))
#                     else:
#                         geo_values = geo_attr_p[t * 3 :t * 3 + 3]
#                     geo_tensor[t, key, :, x, y] = geo_values
#
#     return coord_tensor,col_tensor, geo_tensor, pixel_idx_pairs,para_info
#
#
# def extract_attributes(
#         tensor_coo: torch.Tensor,
#         tensor_col: torch.Tensor,
#         tnesor_g: torch.Tensor,
#         pixel_idx_pairs,
#         axis: int,
#         info: pd.DataFrame
# ) -> torch.Tensor:
#     """
#     使用向量化操作从张量中提取新的属性
#
#     参数:
#     tensor: shape (time_steps, dim1, 3, dim2, dim3) 的张量
#     pixel_idx_pairs: 像素坐标和索引对的列表 [(pixel_coords, idx), ...]
#     axis: 处理的轴 (0, 1, 或 2)
#     old_att: 原始属性张量，shape
#
#     返回:
#     torch.Tensor: 更新后的属性张量
#     """
#
#     time_steps_c2, *dims = tensor_col.shape
#     time_steps_g, *dims = tnesor_g.shape
#     num_pixels = len(pixel_idx_pairs)
#
#     # 将pixel_idx_pairs转换为张量形式以便批处理
#     pixels = torch.tensor([p[0] for p in pixel_idx_pairs])
#     indices = torch.tensor([p[1] for p in pixel_idx_pairs])
#
#     # 创建输出张量，复制原始属性
#     # output_attributes = old_attr.clone()
#     # output_attributes = torch.cat((output_attributes, torch.zeros((output_attributes.shape[0], 1))),dim=1)
#     output_attributes = torch.zeros((num_pixels, 60), dtype=torch.float32)
#     # output_attributes = torch.zeros((num_pixels, 6 + time_steps * 3), dtype=torch.float32)
#     # 根据不同的轴创建索引张量
#     if axis == 0:
#         batch_indices = pixels[:, 0]
#         height_indices = pixels[:, 1]
#         width_indices = pixels[:, 2]
#     elif axis == 1:
#         batch_indices = pixels[:, 1]
#         height_indices = pixels[:, 0]
#         width_indices = pixels[:, 2]
#     else:  # axis == 2
#         batch_indices = pixels[:, 2]
#         height_indices = pixels[:, 0]
#         width_indices = pixels[:, 1]
#
#     # 为每个时间步骤提取颜色值
#     coord_values = tensor_coo[:, batch_indices, :, height_indices, width_indices]
#     output_attributes[indices, :3] = torch.tensor(coord_values.astype(np.int32),dtype=torch.float32).squeeze(1)
#     for t in tqdm(range(time_steps_c2)):
#         # 使用高级索引一次性提取所有像素的颜色值
#         color_values = tensor_col[t, batch_indices, :, height_indices, width_indices]
#
#         # 更新输出属性张量
#         output_attributes[indices, 3 + t * 3:3 + (t + 1) * 3] = torch.tensor(color_values.astype(np.int32),dtype=torch.float32)
#
#         if t < time_steps_g:
#             # 使用高级索引一次性提取所有像素的颜色值
#             geo_values = tnesor_g[t, batch_indices, :, height_indices, width_indices]
#             # 更新输出属性张量
#             output_attributes[indices, 51 + t * 3:51 + (t + 1) * 3] = torch.tensor(geo_values.astype(np.int32),dtype=torch.float32)
#
#     q_attr = output_attributes[:, :59]
#     new_attr = inverse_quantization(q_attr, info)
#     return new_attr


import torch
import pandas as pd
import numpy as np
import pickle

from torch.onnx.symbolic_opset9 import tensor


def log_transform(coords):
    positive = coords > 0
    negative = coords < 0
    zero = coords == 0

    transformed_coords = np.zeros_like(coords)
    transformed_coords[positive] = np.log1p(coords[positive])
    transformed_coords[negative] = -np.log1p(-coords[negative])
    # For zero, no change is needed as transformed_coords is already initialized to zeros

    return transformed_coords

def inverse_log_transform(transformed_coords):
    positive = transformed_coords > 0
    negative = transformed_coords < 0
    zero = transformed_coords == 0

    original_coords = np.zeros_like(transformed_coords)
    original_coords[positive] = np.expm1(transformed_coords[positive])
    original_coords[negative] = -np.expm1(-transformed_coords[negative])
    # For zero, no change is needed as original_coords is already initialized to zeros

    return original_coords

def quantization(attributes, enable=True):
    """
    对四维张量进行分段量化

    Args:
        attributes (torch.Tensor): 输入的四维张量，第一维度为59
        enable (bool, optional): 是否启用量化. Defaults to True.

    Returns:
        tuple: 量化后的张量和信息表
    """
    if not enable:
        return attributes

    # 初始化信息表
    info = pd.DataFrame(columns=['min_value', 'max_value'])

    def single_q(attr, info_table):
        """单个属性的量化处理"""
        # 将属性转换为numpy数组以处理
        attr_np = attr.numpy()
        min_values, max_values = np.min(attr_np), np.max(attr_np)

        # 防止除零错误
        scale = max_values - min_values
        scale = 1 if scale == 0 else scale

        # 执行量化
        q_attr = torch.clamp(
            ((torch.from_numpy(attr_np) - min_values) * 65535 / scale),
            0,
            65535
        ).to(torch.int32)

        # 记录最小值和最大值
        info_table.loc[len(info_table)] = min_values, max_values

        return q_attr, info_table

    # 按指定维度分割和处理张量
    ex_attr_xyz = attributes[:3, :, :, :]
    ex_attr_f = attributes[3:6, :, :, :]
    ex_attr_rest = attributes[6:6 + 45, :, :, :]
    ex_attr_opa = attributes[6 + 45:6 + 45 + 1, :, :, :]
    ex_attr_scale = attributes[6 + 45 + 1:6 + 45 + 4, :, :, :]
    ex_attr_rot = attributes[6 + 45 + 4:6 + 45 + 8, :, :, :]

    # 对不同部分应用对数变换和量化
    t_attr_xyz = torch.Tensor(log_transform(ex_attr_xyz))
    min_values, max_values = torch.min(t_attr_xyz), torch.max(t_attr_xyz)
    info.loc[len(info)] = min_values.numpy(), max_values.numpy()

    q_attr_xyz = torch.clamp(
        ((t_attr_xyz - min_values) * 65535 / (max_values - min_values)),
        0,
        65535
    ).to(torch.int32)

    # 逐部分量化
    q_attr_fdc, info = single_q(ex_attr_f, info)
    q_attr_rest, info = single_q(ex_attr_rest, info)
    q_attr_opa, info = single_q(ex_attr_opa, info)
    q_attr_scale, info = single_q(ex_attr_scale, info)
    q_attr_rot, info = single_q(ex_attr_rot, info)

    # 合并不同部分的量化结果
    q_attr_col = torch.cat([q_attr_fdc, q_attr_rest], dim=0)
    q_attr_geo = torch.cat([q_attr_opa, q_attr_scale, q_attr_rot], dim=0)

    return q_attr_xyz, q_attr_col, q_attr_geo, info

def buildVideoTensor(tensor):
    c,n,h,w = tensor.shape
    xyz, col, geo, info = quantization(tensor)
    geo = torch.cat((geo, torch.zeros((1, n, h, w), dtype=torch.float32)), dim=0)
    a,b,c,d = xyz.shape
    a1 = xyz.reshape(1,b,3,c,d)
    a,b,c,d = col.shape
    a2 = col.reshape(16,b,3,c,d)
    a,b,c,d = geo.shape
    a3 = geo.reshape(3,b,3,c,d)
    return a1,a2,a3,info

def inverse_quantization(q_attr,info):
    min_values, max_values = info.loc[0]
    t_attr_xyz = q_attr[:3,:] * (max_values - min_values) / 65535 + min_values
    ex_attr_xyz = inverse_log_transform(t_attr_xyz)
    ex_attr_xyz = torch.from_numpy(ex_attr_xyz)
    min_values, max_values = info.loc[1]
    ex_attr_f = q_attr[3:6,:] * (max_values - min_values) / 65535 + min_values
    min_values, max_values = info.loc[2]
    ex_attr_rest = q_attr[6:6 + 45,:] * (max_values - min_values) / 65535 + min_values
    min_values, max_values = info.loc[3]
    ex_attr_opa = q_attr[6 + 45,:] * (max_values - min_values) / 65535 + min_values
    ex_attr_opa = ex_attr_opa.unsqueeze(0)
    min_values, max_values = info.loc[4]
    ex_attr_scale = q_attr[6 + 45 + 1:6 + 45 + 4,:] * (max_values - min_values) / 65535 + min_values
    min_values, max_values = info.loc[5]
    ex_attr_rot = q_attr[6 + 45 + 4:6 + 45 + 8,:] * (max_values - min_values) / 65535 + min_values
    new_attr = torch.cat([ex_attr_xyz, ex_attr_f, ex_attr_rest, ex_attr_opa, ex_attr_scale, ex_attr_rot], dim=0)
    return new_attr


def extract_attributes(tensor1, tensor2, tensor3, info):
    """
    重塑和排列三个输入张量

    参数:
    tensor1: 形状为 (1, n, 3, m, l)
    tensor2: 形状为 (16, n, 3, m, l)
    tensor3: 形状为 (3, n, 3, m, l)

    返回:
    重塑后的张量，形状为 (n*m*l, 1*3 + 16*3 + 3*3)
    """
    if not isinstance(tensor1, torch.Tensor):
        tensor1 = torch.from_numpy(tensor1.astype(np.int32))
    if not isinstance(tensor2, torch.Tensor):
        tensor2 = torch.from_numpy(tensor2.astype(np.int32))
    if not isinstance(tensor3, torch.Tensor):
        tensor3 = torch.from_numpy(tensor3.astype(np.int32))
    # 首先确认三个张量的形状匹配
    n, m, l = tensor1.shape[1], tensor1.shape[3], tensor1.shape[4]

    # 调整张量维度顺序，使其最后变为 (n, 3, m, l, ...)
    tensor1_reshaped = tensor1.reshape(3, n, m, l)
    tensor2_reshaped = tensor2.reshape(48, n, m, l)
    tensor3_reshaped = tensor3.reshape(9, n, m, l)
    tensor3_reshaped = tensor3_reshaped[:8]
    # 沿第二个维度拼接
    result = torch.cat([tensor1_reshaped, tensor2_reshaped, tensor3_reshaped], dim=0)
    result = inverse_quantization(result, info)

    return result
