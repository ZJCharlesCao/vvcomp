from collections import namedtuple
import numpy as np
from plyfile import PlyData, PlyElement
import copy
import hashlib
from collections import defaultdict
from seg import compute_point_cloud_bounds, PointKDTree, get_index_and_ranges
import torch
import os
from glob import glob
from tqdm import tqdm




# def build_video(axis,lis,attibutes,ranges):
#     result = defaultdict(lambda: {'pixel': [], 'idx': []})
#     if axis == 0:
#         # 遍历矩阵
#         for p, idx in lis:
#             result[p[0]]['pixel'].append([p[1],p[2]])
#             result[p[0]]['idx'].append(idx)
#         pre_tensor = torch.empty(round(48/3),ranges[0][1],3,ranges[1][1],ranges[2][1],dtype=torch.float32)
#
#     elif axis == 1:
#         for p, idx in lis:
#             result[p[1]]['pixel'].append([p[0], p[2]])
#             result[p[1]]['idx'].append(idx)
#         pre_tensor = torch.empty(round(48/3),ranges[1][1],3,ranges[0][1],ranges[2][1],dtype=torch.float32)
#
#     elif axis == 2:
#         for p, idx in lis:
#             result[p[2]]['pixel'].append([p[0], p[1]])
#             result[p[2]]['idx'].append(idx)
#         pre_tensor = torch.empty(round(48/3),ranges[2][1],3, ranges[0][1],ranges[1][1],dtype=torch.float32)
#
#
#     for key in result.keys():
#         for j in range(len(result[key]['idx'])):
#             mark = attibutes[result[key]['idx'][j]]
#             for i in range(round(48 / 3)):
#                 k = torch.tensor(list(mark)[i * 3 + 6:i * 3 + 9])
#                 pre_tensor[i,key,:,result[key]['pixel'][j][0],result[key]['pixel'][j][1]] = k
#     return pre_tensor

import torch
from collections import defaultdict


def build_video(axis, pixel_idx_pairs, attributes, ranges):
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
    # 确定输出张量的形状
    tensor_shapes = {
        0: (round(48 / 3), ranges[0][1], 3, ranges[1][1], ranges[2][1]),
        1: (round(48 / 3), ranges[1][1], 3, ranges[0][1], ranges[2][1]),
        2: (round(48 / 3), ranges[2][1], 3, ranges[0][1], ranges[1][1])
    }

    # 初始化输出张量
    output_tensor = torch.zeros(tensor_shapes[axis], dtype=torch.float32)

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
    time_steps = round(48 / 3)
    for key, pixels in pixel_map.items():
        for x, y, idx in pixels:
            attribute = attributes[idx]
            for t in range(time_steps):
                color_values = attribute[t * 3 + 3:t * 3 + 6]
                if axis == 0:
                    output_tensor[t, key, :, x, y] = color_values
                elif axis == 1:
                    output_tensor[t, key, :, x, y] = color_values
                else:  # axis == 2
                    output_tensor[t, key, :, x, y] = color_values

    return output_tensor


# def extract_attributes(tensor, pixel_idx_pairs, axis,old_att):
#     """
#     从张量中提取新的属性
#
#     参数:
#     tensor: torch.Tensor - 输入视频张量
#     pixel_idx_pairs: list of tuples - 像素坐标和索引对的列表
#     axis: int - 处理的轴 (0, 1, 或 2)
#
#     返回:
#     torch.Tensor - 新的属性张量
#     """
#     time_steps, _, channels, *_ = tensor.shape
#     num_pixels = len(pixel_idx_pairs)
#
#     # 初始化输出属性张量 (前6个值预留为0)
#     output_attributes = old_att
#
#     # 创建像素到索引的映射
#     pixel_to_idx = {tuple(p): i for p, i in pixel_idx_pairs}
#
#     # 提取属性
#     for t in range(time_steps):
#         for p, idx in tqdm(pixel_idx_pairs):
#             if axis == 0:
#                 color_values = tensor[t, p[0], :, p[1], p[2]]
#             elif axis == 1:
#                 color_values = tensor[t, p[1], :, p[0], p[2]]
#             else:  # axis == 2
#                 color_values = tensor[t, p[2], :, p[0], p[1]]
#
#             output_attributes[idx, 6 + t * channels:6 + (t + 1) * channels] = torch.tensor(color_values, dtype=torch.float32)
#
#     return output_attributes

def extract_attributes(
        tensor: torch.Tensor,
        pixel_idx_pairs,
        axis: int,
        old_att: torch.Tensor
) -> torch.Tensor:
    """
    使用向量化操作从张量中提取新的属性

    参数:
    tensor: shape (time_steps, dim1, 3, dim2, dim3) 的张量
    pixel_idx_pairs: 像素坐标和索引对的列表 [(pixel_coords, idx), ...]
    axis: 处理的轴 (0, 1, 或 2)
    old_att: 原始属性张量，shape (num_pixels, 6 + time_steps * 3)

    返回:
    torch.Tensor: 更新后的属性张量
    """
    time_steps, *dims = tensor.shape
    num_pixels = len(pixel_idx_pairs)

    # 将pixel_idx_pairs转换为张量形式以便批处理
    pixels = torch.tensor([p[0] for p in pixel_idx_pairs])
    indices = torch.tensor([p[1] for p in pixel_idx_pairs])

    # 创建输出张量，复制原始属性
    output_attributes = old_att.clone()
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
    for t in tqdm(range(time_steps)):
        # 使用高级索引一次性提取所有像素的颜色值
        color_values = tensor[t, batch_indices, :, height_indices, width_indices]

        # 更新输出属性张量
        output_attributes[indices, 3 + t * 3:3 + (t + 1) * 3] = torch.tensor(color_values,dtype=torch.float32)

    return output_attributes

def save_yuv420p_video(yuv_tensor, output_path):
    """
    Save YUV tensor data as a raw YUV420p video file.

    Args:
    yuv_tensor (torch.Tensor): 5D tensor in YUV format with shape
                               (num_videos, num_frames, 3, height, width)
    output_path (str): Base path to save the output videos
    """
    num_videos, num_frames, _, height, width  = yuv_tensor.shape

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for video_idx in range(num_videos):
        video_path = f"{output_path}_{video_idx}.yuv"
        with open(video_path, 'wb') as f:
            for frame_idx in range(num_frames):
                # Extract Y plane
                y_plane = yuv_tensor[video_idx, frame_idx, 0].numpy()
                # Extract U plane and ensure correct size (height and width are half of Y)
                u_plane = yuv_tensor[video_idx, frame_idx, 1, ::2, ::2].numpy()
                # Extract V plane and ensure correct size (height and width are half of Y)
                v_plane = yuv_tensor[video_idx, frame_idx, 2, ::2, ::2].numpy()

                # Write data in YUV420p format
                f.write(y_plane.tobytes())
                f.write(u_plane.tobytes())
                f.write(v_plane.tobytes())

        print(f"Saved video {video_idx} to {video_path}")
    return width, height, num_frames


# def read_yuv420p_video(file_path, width, height, num_frames=None):
#     frame_size = width * height * 3 // 2
#     with open(file_path, 'rb') as f:
#         video_data = f.read()
#
#     video_tensor = torch.zeros((num_frames, 3, height, width), dtype=torch.float32)
#
#     for i in range(num_frames):
#         frame_data = video_data[i * frame_size:(i + 1) * frame_size]
#         y_plane = np.frombuffer(frame_data[:width * height], dtype=np.uint8).reshape(height, width)
#         u_plane = np.frombuffer(frame_data[width * height:width * height + width * height // 4],
#                                 dtype=np.uint8).reshape(height // 2, width // 2)
#         v_plane = np.frombuffer(frame_data[width * height + width * height // 4:], dtype=np.uint8).reshape(height // 2,
#                                                                                                            width // 2)
#
#         u_upscaled = np.repeat(np.repeat(u_plane, 2, axis=0), 2, axis=1)
#         v_upscaled = np.repeat(np.repeat(v_plane, 2, axis=0), 2, axis=1)
#
#         video_tensor[i, 0] = torch.from_numpy(y_plane)
#         video_tensor[i, 1] = torch.from_numpy(u_upscaled)
#         video_tensor[i, 2] = torch.from_numpy(v_upscaled)
#
#     return video_tensor
#
#
# def yuv420p_videos_to_tensor(video_dir, width, height, file_pattern="*.yuv", num_frames=None):
#     video_files = glob(os.path.join(video_dir, file_pattern))
#     if not video_files:
#         raise ValueError(f"No video files found in {video_dir} matching pattern {file_pattern}")
#
#     first_video = read_yuv420p_video(video_files[0], width, height, num_frames)
#     actual_num_frames = first_video.shape[0]
#
#     all_videos_tensor = torch.zeros((len(video_files), actual_num_frames, 3, height, width), dtype=torch.float32)
#
#     for i, video_file in enumerate(video_files, 1):
#         print(video_file)
#         video_tensor = read_yuv420p_video(video_file, width, height, actual_num_frames)
#         all_videos_tensor[i] = video_tensor
#
#
#     return all_videos_tensor, video_files