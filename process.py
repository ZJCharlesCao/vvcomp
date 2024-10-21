import torch
from plyfile import PlyData, PlyElement
import transform,seg
import glob

import numpy as np
from typing import Tuple, Optional, List
import os
from pathlib import Path


class YUVVideoWriter:
    def __init__(self, filename: str, frame_width: int, frame_height: int):
        self.filename = filename
        self.width = frame_width
        self.height = frame_height
        self.file = open(filename, 'wb')

    def write_frame(self, y: np.ndarray, u: np.ndarray, v: np.ndarray):
        self.file.write(y.tobytes())
        self.file.write(u.tobytes())
        self.file.write(v.tobytes())

    def close(self):
        self.file.close()


class YUVVideoReader:
    def __init__(self, filename: str, frame_width: int, frame_height: int):
        self.filename = filename
        self.width = frame_width
        self.height = frame_height
        self.file = open(filename, 'rb')
        self.frame_size = (self.height * self.width * 3) // 2  # YUV420p大小

    def read_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        try:
            y_size = self.height * self.width
            uv_size = y_size // 4

            y_data = self.file.read(y_size)
            u_data = self.file.read(uv_size)
            v_data = self.file.read(uv_size)

            if len(y_data) < y_size:  # 文件结束
                return None

            y = np.frombuffer(y_data, dtype=np.uint8).reshape((self.height, self.width))
            u = np.frombuffer(u_data, dtype=np.uint8).reshape((self.height // 2, self.width // 2))
            v = np.frombuffer(v_data, dtype=np.uint8).reshape((self.height // 2, self.width // 2))

            return y, u, v
        except:
            return None

    def close(self):
        self.file.close()


def downsample_uv(plane: np.ndarray) -> np.ndarray:
    """对UV平面进行2x2降采样"""
    return plane[::2, ::2]


def upsample_uv(plane: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """对UV平面进行上采样"""
    return np.repeat(np.repeat(plane, 2, axis=0), 2, axis=1)


def tensor_to_yuv420p_videos(tensor: np.ndarray, output_dir: str) -> List[str]:
    """
    将5D张量转换为多个YUV420P格式的视频

    参数:
    tensor: 5D numpy数组，shape为(num_videos, num_frames, channels=3, height, width)
    output_dir: 输出视频文件的目录路径

    返回:
    List[str]: 生成的视频文件路径列表
    """
    try:
        num_videos, num_frames, channels, height, width = tensor.shape
        assert channels == 3, "输入张量必须是3通道(YUV)"

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        output_files = []
        for vid in range(num_videos):
            output_path = os.path.join(output_dir, f"video_{vid}.yuv")
            writer = YUVVideoWriter(output_path, width, height)

            for frame in range(num_frames):
                y_plane = tensor[vid, frame, 0]
                u_plane = downsample_uv(tensor[vid, frame, 1])
                v_plane = downsample_uv(tensor[vid, frame, 2])

                writer.write_frame(y_plane, u_plane, v_plane)

            writer.close()
            output_files.append(output_path)

        return output_files

    except Exception as e:
        print(f"转换过程中出错: {str(e)}")
        return []


def yuv420p_videos_to_tensor(video_paths: List[str], frames_per_video: int, height: int, width: int) -> Optional[
    np.ndarray]:
    """
    将多个YUV420P格式的视频转换回5D张量

    参数:
    video_paths: YUV视频文件路径列表
    frames_per_video: 每个视频的帧数
    height: 视频高度
    width: 视频宽度

    返回:
    numpy数组: 5D张量，shape为(num_videos, frames_per_video, channels=3, height, width)
    """
    try:
        num_videos = len(video_paths)
        tensor = np.zeros((num_videos, frames_per_video, 3, height, width), dtype=np.uint8)

        for vid, video_path in enumerate(video_paths):
            reader = YUVVideoReader(video_path, width, height)

            for frame_idx in range(frames_per_video):
                frame_data = reader.read_frame()
                if frame_data is None:
                    break

                y, u, v = frame_data
                tensor[vid, frame_idx, 0] = y
                tensor[vid, frame_idx, 1] = upsample_uv(u, height, width)
                tensor[vid, frame_idx, 2] = upsample_uv(v, height, width)

            reader.close()

        return tensor

    except Exception as e:
        print(f"转换过程中出错: {str(e)}")
        return None


def tensor_to_rgb24_videos(tensor: np.ndarray, output_dir: str) :
    """
    将5D张量转换为多个RGB24格式的视频
    参数:
    tensor: 5D numpy数组，shape为(num_videos, num_frames, channels=3, height, width)
    output_dir: 输出视频文件的目录路径
    返回:
    List[str]: 生成的视频文件路径列表
    """
    try:
        num_videos, num_frames, channels, height, width = tensor.shape
        assert channels == 3, "输入张量必须是3通道(RGB)"

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        output_files = []
        for vid in range(num_videos):
            output_path = os.path.join(output_dir, f"video_{vid}.rgb")
            with open(output_path, 'wb') as writer:
                for frame in range(num_frames):
                    # RGB数据直接写入，无需转换
                    frame_data = tensor[vid, frame].transpose(1, 2, 0).tobytes()
                    writer.write(frame_data)
            output_files.append(output_path)

        return output_files
    except Exception as e:
        print(f"转换过程中出错: {str(e)}")
        return []


def rgb24_videos_to_tensor(video_paths, frames_per_video: int, height: int, width: int) -> Optional[
    np.ndarray]:
    """
    将多个RGB24格式的视频转换回5D张量
    参数:
    video_paths: RGB视频文件路径列表
    frames_per_video: 每个视频的帧数
    height: 视频高度
    width: 视频宽度
    返回:
    numpy数组: 5D张量，shape为(num_videos, frames_per_video, channels=3, height, width)
    """
    try:
        num_videos = len(video_paths)
        tensor = np.zeros((num_videos, frames_per_video, 3, height, width), dtype=np.uint8)
        frame_size = height * width * 3  # RGB24每帧的字节数

        for vid, video_path in enumerate(video_paths):
            with open(video_path, 'rb') as reader:
                for frame_idx in range(frames_per_video):
                    frame_data = reader.read(frame_size)
                    if not frame_data or len(frame_data) < frame_size:
                        break

                    # 将字节数据转换为numpy数组并重塑
                    frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape(height, width, 3)
                    # 转置以匹配所需的格式 (3, height, width)
                    tensor[vid, frame_idx] = frame_array.transpose(2, 0, 1)

        return tensor
    except Exception as e:
        print(f"转换过程中出错: {str(e)}")
        return None
