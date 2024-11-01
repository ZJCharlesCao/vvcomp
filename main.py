import process,seg
import os
import transform
import torch
import numpy as np
from plyfile import PlyData, PlyElement



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

def update_file_paths(file_paths: list,a) -> list:
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
        new_path = os.path.join(directory+'/iteration_'+str(a), file_name)
        updated_paths.append(new_path)
    return updated_paths

if __name__ == '__main__':
    axis = 0
    num_videos = 16
    frames_per_video = 160
    height, width = 70, 150
    outdir = '/data2/zijian/videocomp/main/output_videos'
    inputply = '/data2/zijian/videocomp/main/point_cloud.ply'

    data = PlyData.read(inputply)
    points = np.vstack([data["vertex"]["x"], data["vertex"]["y"], data["vertex"]["z"]]).T
    # old_attributes = np.array(data["vertex"]).T
    gs = gaussianmodel()
    old_attributes = gs.load_ply(inputply)

    space_range = [[0, 160], [0, 70], [0, 150]]
    point_range = transform.compute_point_cloud_bounds(points)
    print(point_range)
    tree = seg.PointKDTree(points, point_range, space_range)
    codebook = seg.get_cb(tree.root)
    videobook = transform.build_video(axis, codebook, old_attributes, space_range)
    # torch.save(videobook, 'output/video.pt')
    # x = torch.load('output/video.pt').numpy()
    x = videobook.numpy()
    x_min, x_max = np.min(x), np.max(x)

    # 量化：将浮点数映射到 0-255 的 uint8
    sample_tensor = np.clip(((x - x_min) * 65535 / (x_max - x_min)), 0, 65535).astype(np.uint16)
    output_files = process.tensor_to_rgb48_videos(sample_tensor, outdir)

    if output_files:
        # print(f"成功生成 {len(output_files)} 个视频文件:")
        for file_path in output_files:
            print(f"  - {file_path}")

        # # 将视频转换回张量
        recovered_tensor_0 = process.rgb48_videos_to_tensor(output_files, frames_per_video, height, width)
        # 反量化：将 uint8 重新映射回原始范围
        x_reconstructed_0 = recovered_tensor_0 / 65535 * (x_max - x_min) + x_min
        new_attribute1 = transform.extract_attributes(x_reconstructed_0, codebook, axis, old_attributes)

        for i in range(1, 11):
            os.makedirs(outdir+'/iteration_'+str(i), exist_ok=True)
            out = outdir+'/iteration_'+str(i)
            os.system(' python /data2/zijian/videocomp/main/mpegcomp.py --input_dir /data2/zijian/videocomp/main/output_videos --output_dir '+out+' --width 150 --height 70 --pattern *.rgb48le --pixel_format rgb48le --qp '+str(i))
            new_path = update_file_paths(output_files,i)
            recovered_tensor = process.rgb48_videos_to_tensor(new_path, frames_per_video, height, width)
            x_reconstructed = recovered_tensor / 65535 * (x_max-x_min) +x_min
            new_attribute2 = transform.extract_attributes(x_reconstructed, codebook, axis, old_attributes)
            gs.save_ply(new_attribute2, out + "/point_cloud.ply")
        gs.save_ply(new_attribute1, outdir + "/point_cloud_0.ply")

    # print(x_reconstructed)
