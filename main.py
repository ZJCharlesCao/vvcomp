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


# def calculate_yuv(original: np.ndarray, recovered: np.ndarray) -> float:
#     """计算原始张量和恢复张量之间的PSNR"""
#     mse = np.mean((original - recovered) ** 2)
#     if mse == 0:
#         return float('inf')
#     max_pixel = 255.0
#     return 20 * np.log10(max_pixel / np.sqrt(mse))

def update_file_paths(file_paths: list) -> list:
    """
    将字符串列表每个元素的前半部分改为new_yuv

    参数:
    file_paths: 文件路径列表

    返回:
    List[str]: 修改后的文件路径列表
    """
    updated_paths = []
    for path in file_paths:
        directory, file_name = os.path.split(path)
        new_path = os.path.join('new_yuv', file_name)
        updated_paths.append(new_path)
    return updated_paths

class gaussianmodel:
    def __init__(self):
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._opacity = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
    def load_ply(self,path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(3 + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


        self._xyz = torch.tensor(xyz, dtype=torch.float)
        self._features_dc = torch.tensor(features_dc, dtype=torch.float).transpose(1, 2)
        self._features_rest = torch.tensor(features_extra, dtype=torch.float).transpose(1, 2)
        f_dc = self._features_dc.transpose(1, 2).flatten(start_dim=1)
        f_rest = self._features_rest.transpose(1, 2).flatten(start_dim=1)
        self._opacity = torch.tensor(opacities, dtype=torch.float)
        self._scaling = torch.tensor(scales, dtype=torch.float)
        self._rotation = torch.tensor(rots, dtype=torch.float)
        att = torch.cat([self._xyz,f_dc, f_rest,self._opacity,self._scaling, self._rotation], dim=1)
        return att

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, tensor,path):

        xyz = self._xyz.numpy()
        normals = np.zeros_like(xyz)
        f_dc = tensor[:, 3:6].numpy()
        f_rest = tensor[:, 6:6 + 3 * (3 + 1) ** 2 - 3].numpy()
        opacities = self._opacity.numpy()
        scale = self._scaling.numpy()
        rotation = self._rotation.numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

if __name__ == '__main__':


    axis = 0
    data = PlyData.read("/data2/zijian/videocomp/Laplace/point_cloud.ply")
    points = np.vstack([data["vertex"]["x"], data["vertex"]["y"], data["vertex"]["z"]]).T
    # old_attributes = np.array(data["vertex"]).T
    gs = gaussianmodel()
    old_attributes = gs.load_ply("/data2/zijian/videocomp/Laplace/point_cloud.ply")
    space_range = [[0, 160], [0, 70], [0, 150]]
    point_range = transform.compute_point_cloud_bounds(points)
    print(point_range)
    tree = seg.PointKDTree(points, point_range, space_range)
    indexs_and_ranges = transform.get_index_and_ranges(tree.root)
    videobook = transform.build_video(axis, indexs_and_ranges, old_attributes, space_range)
    # torch.save(videobook, 'output/video.pt')
    # x = torch.load('output/video.pt').numpy()
    x = videobook.numpy()

    num_videos = 16
    frames_per_video = 160
    height, width = 70, 150

    # sample_tensor = np.random.randint(0, 256,
    #                                   (num_videos, frames_per_video, 3, height, width),
    #                                   dtype=np.uint8)
    # 生成一个随机的浮点数张量，范围在 [-3, 15] 之间
    x_min, x_max = -3, 15
    # x = np.random.uniform(x_min, x_max, (num_videos, frames_per_video, 3, height, width))

    # 量化：将浮点数映射到 0-255 的 uint8
    sample_tensor = np.clip(((x - x_min) * 255 / (x_max - x_min)), 0, 255).astype(np.uint8)

    # 保存为多个视频文件
    output_dir = "output_videos"
    output_files = tensor_to_yuv420p_videos(sample_tensor, output_dir)

    if output_files:
        print(f"成功生成 {len(output_files)} 个视频文件:")
        for file_path in output_files:
            print(f"  - {file_path}")

        # 将视频转换回张量
        recovered_tensor = yuv420p_videos_to_tensor(output_files, frames_per_video, height, width)
        # 反量化：将 uint8 重新映射回原始范围
        x_reconstructed = recovered_tensor / 255 * 18 - 3
        # new_path = update_file_paths(output_files)
        # recovered_tensor = yuv420p_videos_to_tensor(new_path, frames_per_video, height, width)
        # x_reconstructed = recovered_tensor / 255 * 18 - 3
        new_attribute = transform.extract_attributes(x_reconstructed, indexs_and_ranges, axis, old_attributes)
    gs.save_ply(new_attribute,output_dir+"/point_cloud.ply")

    # print(x_reconstructed)
