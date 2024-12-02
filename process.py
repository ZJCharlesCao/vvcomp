import torch
from plyfile import PlyData, PlyElement
import transform,seg
import glob

import numpy as np
from typing import Tuple, Optional, List
import os
from pathlib import Path

# def tensor_to_rgb24_videos(tensor: np.ndarray, output_dir: str) :
#     """
#     将5D张量转换为多个RGB24格式的视频
#     参数:
#     tensor: 5D numpy数组，shape为(num_videos, num_frames, channels=3, height, width)
#     output_dir: 输出视频文件的目录路径
#     返回:
#     List[str]: 生成的视频文件路径列表
#     """
#     try:
#         num_videos, num_frames, channels, height, width = tensor.shape
#         assert channels == 3, "输入张量必须是3通道(RGB)"
#
#         # 确保输出目录存在
#         os.makedirs(output_dir, exist_ok=True)
#
#         output_files = []
#         for vid in range(num_videos):
#             output_path = os.path.join(output_dir, f"video_{vid}.rgb")
#             with open(output_path, 'wb') as writer:
#                 for frame in range(num_frames):
#                     # RGB数据直接写入，无需转换
#                     frame_data = tensor[vid, frame].transpose(1, 2, 0).tobytes()
#                     writer.write(frame_data)
#             output_files.append(output_path)
#
#         return output_files
#     except Exception as e:
#         print(f"转换过程中出错: {str(e)}")
#         return []
#
#
# def rgb24_videos_to_tensor(video_paths, frames_per_video: int, height: int, width: int) -> Optional[
#     np.ndarray]:
#     """
#     将多个RGB24格式的视频转换回5D张量
#     参数:
#     video_paths: RGB视频文件路径列表
#     frames_per_video: 每个视频的帧数
#     height: 视频高度
#     width: 视频宽度
#     返回:
#     numpy数组: 5D张量，shape为(num_videos, frames_per_video, channels=3, height, width)
#     """
#     try:
#         num_videos = len(video_paths)
#         tensor = np.zeros((num_videos, frames_per_video, 3, height, width), dtype=np.uint8)
#         frame_size = height * width * 3  # RGB24每帧的字节数
#
#         for vid, video_path in enumerate(video_paths):
#             with open(video_path, 'rb') as reader:
#                 for frame_idx in range(frames_per_video):
#                     frame_data = reader.read(frame_size)
#                     if not frame_data or len(frame_data) < frame_size:
#                         break
#
#                     # 将字节数据转换为numpy数组并重塑
#                     frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape(height, width, 3)
#                     # 转置以匹配所需的格式 (3, height, width)
#                     tensor[vid, frame_idx] = frame_array.transpose(2, 0, 1)
#
#         return tensor
#     except Exception as e:
#         print(f"转换过程中出错: {str(e)}")
#         return None


def tensor_to_rgb48(tensor: np.ndarray, output_dir: str) -> List[str]:
    """
    将5D张量转换为多个RGB48格式的视频文件

    参数:
    tensor: 5D numpy数组，shape为(num_videos, num_frames, channels=3, height, width)
    output_dir: 输出视频文件的目录路径

    返回:
    List[str]: 生成的视频文件路径列表
    """
    try:
        num_videos, num_frames, channels, height, width = tensor.shape
        assert channels == 3, "输入张量必须是3通道(RGB)"
        frame_size = height * width * 3 * 2  # RGB48每帧的字节数

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        for vid in range(num_videos):
            if num_videos == 16:
                output_path = os.path.join(output_dir, f"color_{vid}.rgb48le")
            elif num_videos == 3:
                output_path = os.path.join(output_dir,f"geo_{vid}.rgb48le")
            else:
                output_path = os.path.join(output_dir, f"xyz.rgb48le")
            # 转置并重组数据 (num_frames, channels, height, width) -> (num_frames, height, width, channels)
            video_data = tensor[vid].transpose(0, 2, 3, 1)
            # 确保数据是小端序uint16
            video_data = video_data.astype('<u2')
            # 将所有帧写入单个文件
            with open(output_path, 'wb') as f:
                f.write(video_data.tobytes())
            output_files.append(output_path)

        return output_files
    except Exception as e:
        print(f"转换过程中出错: {str(e)}")
        return []


def rgb48_videos_to_tensor(video_paths: List[str], frames_per_video: int, height: int, width: int) -> Optional[
    np.ndarray]:
    """
    将多个RGB48格式的视频文件转换回5D张量

    参数:
    video_paths: RGB48视频文件路径列表
    frames_per_video: 每个视频的帧数
    height: 视频高度
    width: 视频宽度

    返回:
    numpy数组: 5D张量，shape为(num_videos, frames_per_video, channels=3, height, width)
    """
    try:
        num_videos = len(video_paths)
        tensor = np.zeros((num_videos, frames_per_video, 3, height, width), dtype='<u2')
        frame_size = height * width * 3 * 2  # RGB48每帧的字节数

        for vid, video_path in enumerate(video_paths):
            with open(video_path, 'rb') as f:
                # 读取整个视频文件
                video_data = f.read()
                # 确保数据完整性
                expected_size = frames_per_video * frame_size
                if len(video_data) != expected_size:
                    raise ValueError(f"视频文件 {video_path} 大小不符合预期")

                # 将字节数据转换为numpy数组并重塑
                video_array = np.frombuffer(video_data, dtype='<u2').reshape(
                    frames_per_video, height, width, 3)
                # 转置以匹配所需的输出格式 (frames, channels, height, width)
                tensor[vid] = video_array.transpose(0, 3, 1, 2)

        return tensor
    except Exception as e:
        print(f"转换过程中出错: {str(e)}")
        return None