from numpy.distutils.lib2def import output_def
from scipy.constants import point

import process, transform, mpegcomp
import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement
import pickle



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

        f_dc = torch.clamp(f_dc, -2, 4)
        f_rest = torch.clamp(f_rest, -1, 1)
        self._opacity = torch.clamp(self._opacity, -6,12)
        self._rotation = torch.clamp(self._rotation, -1, 2)
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

        # xyz = self._xyz.numpy()
        xyz = tensor[:, 0:3].numpy()
        normals = np.zeros_like(xyz)

        # opacities = self._opacity.numpy()
        # scale = self._scaling.numpy()
        # rotation = self._rotation.numpy()
        # f_dc =self._features_dc.transpose(1, 2).flatten(start_dim=1).numpy()
        # f_rest = self._features_rest.transpose(1, 2).flatten(start_dim=1).numpy()
        f_dc = tensor[:, 3:6].numpy()
        f_rest = tensor[:, 6:6 + 3 * (3 + 1) ** 2 - 3].numpy()
        opacities = tensor[:, 6 + 3 * (3 + 1) ** 2 - 3:6 + 3 * (3 + 1) ** 2 - 3 + 1].numpy()
        scale = tensor[:, 6 + 3 * (3 + 1) ** 2 - 2:6 + 3 * (3 + 1) ** 2 +1].numpy()
        rotation = tensor[:, 6 + 3 * (3 + 1) ** 2 + 1:6 + 3 * (3 + 1) ** 2 + 5].numpy()

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
        last_slash_index = path.rfind('/')
        # 在最后一个斜杠之前寻找倒数第二个斜杠的位置
        second_last_slash_index = path.rfind('/', 0, last_slash_index)
        dir, filename = path[:second_last_slash_index], path[second_last_slash_index + 1:]
        new_path = os.path.join(dir+'/iteration_'+str(a), filename)
        updated_paths.append(new_path)
    return updated_paths


def main(frame_per_video, height, width, file_path, outdir):
    axis = 0
    # num_videos = 16
    inputply = '/data2/zijian/videocomp/main/input/point_cloud.ply'


    data = PlyData.read(inputply)
    # points = np.vstack([data["vertex"]["x"], data["vertex"]["y"], data["vertex"]["z"]]).T
    gs = gaussianmodel()

    old_attributes = gs.load_ply(inputply)

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    data = torch.from_numpy(data)
    coordbook, colorbook, geobook, info = transform.buildVideoTensor(data)
    output_files_coord = process.tensor_to_rgb48(coordbook.numpy(), outdir+'/coord')
    output_files_col = process.tensor_to_rgb48(colorbook.numpy(), outdir+'/color')
    output_files_geo = process.tensor_to_rgb48(geobook.numpy(), outdir+'/geo')

    if output_files_col is not None and output_files_geo is not None:
        # # 将视频转换回张量
        recovered_tensor_coord = process.rgb48_videos_to_tensor(output_files_coord, frames_per_video, height, width)
        recovered_tensor_col = process.rgb48_videos_to_tensor(output_files_col, frames_per_video, height, width)
        recovered_tensor_geo = process.rgb48_videos_to_tensor(output_files_geo, frames_per_video, height, width)

        new_attributes = transform.extract_attributes(recovered_tensor_coord, recovered_tensor_col, recovered_tensor_geo, info)
        new_attributes = new_attributes.reshape(59, frame_per_video*height*width).transpose(0, 1)

        gs.save_ply(new_attributes, outdir + "/point_cloud_0.ply")

        for i in range(10, 11):
            out = outdir+'/iteration_'+str(i)
            os.makedirs(out, exist_ok=True)
            mpegcomp.process_directory(input_dir=outdir, output_dir=out, height=height, width=width, qp=i)

            new_path_coord = update_file_paths(output_files_coord,i)
            new_path_c = update_file_paths(output_files_col,i)
            new_path_g = update_file_paths(output_files_geo,i)

            compressed_tensor_coord = process.rgb48_videos_to_tensor(new_path_coord, frames_per_video, height, width)
            compressed_tensor_col = process.rgb48_videos_to_tensor(new_path_c, frames_per_video, height, width)
            compressed_tensor_geo = process.rgb48_videos_to_tensor(new_path_g, frames_per_video, height, width)

            compressed_attributes = transform.extract_attributes(compressed_tensor_coord,compressed_tensor_col,compressed_tensor_geo, info)
            compressed_attributes = compressed_attributes.reshape(59, frame_per_video*height*width).transpose(0, 1)
            gs.save_ply(compressed_attributes, out + "/point_cloud.ply")


    # print(x_reconstructed)
if __name__ == '__main__':
    frames_per_video = 96
    height, width = 96, 96
    outdir = '/data2/zijian/videocomp/main/output'
    file_path = '/data2/zijian/videocomp/main/input/4-dimension_v1.pkl'
    main(frames_per_video,height,width, file_path, outdir)