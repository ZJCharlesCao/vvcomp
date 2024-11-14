from scipy.constants import point

import process, transform, mpegcomp
import os
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


if __name__ == '__main__':
    axis = 0
    num_videos = 16
    frames_per_video = 160
    height, width = 70, 150
    outdir = '/data2/zijian/videocomp/main/output'
    inputply = '/data2/zijian/videocomp/main/point_cloud.ply'

    data = PlyData.read(inputply)
    # points = np.vstack([data["vertex"]["x"], data["vertex"]["y"], data["vertex"]["z"]]).T
    gs = gaussianmodel()

    old_attributes = gs.load_ply(inputply)
    ex_attr1 = old_attributes[:, 6:15]
    xyz = old_attributes[:, 0:3]
    # ex_attr = old_attributes[:, 3:]
    # # 获取张量中的最小值和最大值
    # min_a, max_a = torch.min(ex_attr).numpy(), torch.max(ex_attr).numpy()
    # print(max_a, min_a)
    # q_attr = torch.clamp(((ex_attr - min_a) * 65535 / (max_a - min_a)), 0, 65535).to(torch.int32)
    ex_attr_f = old_attributes[:, 3:6]
    ex_attr_rest = old_attributes[:, 6:6 + 3 * (3 + 1) ** 2 - 3]
    ex_attr_opa, ex_attr_scale, ex_attr_rot = old_attributes[:, 6 + 3 * (3 + 1) ** 2 - 3:6 + 3 * (3 + 1) ** 2 - 2], old_attributes[:, 6 + 3 * (3 + 1) ** 2 - 2:6 + 3 * (3 + 1) ** 2 + 1], old_attributes[:, 6 + 3 * (3 + 1) ** 2 + 1:6 + 3 * (3 + 1) ** 2 + 5]

    min_a, max_a = torch.min(ex_attr_f).numpy(), torch.max(ex_attr_f).numpy()
    q_attr_f = torch.clamp(((ex_attr_f - min_a) * 65535 / (max_a - min_a)), 0, 65535).to(torch.int32)
    min_b, max_b = torch.min(ex_attr_rest).numpy(), torch.max(ex_attr_rest).numpy()
    q_attr_rest = torch.clamp(((ex_attr_rest - min_b) * 65535 / (max_b - min_b)), 0, 65535).to(torch.int32)
    min_c, max_c = torch.min(ex_attr_opa).numpy(), torch.max(ex_attr_opa).numpy()
    q_attr_opa = torch.clamp(((ex_attr_opa - min_c) * 65535 / (max_c - min_c)), 0, 65535).to(torch.int32)
    min_d, max_d = torch.min(ex_attr_scale).numpy(), torch.max(ex_attr_scale).numpy()
    q_attr_scale = torch.clamp(((ex_attr_scale - min_d) * 65535 / (max_d - min_d)), 0, 65535).to(torch.int32)
    min_e, max_e = torch.min(ex_attr_rot).numpy(), torch.max(ex_attr_rot).numpy()
    q_attr_rot = torch.clamp(((ex_attr_rot - min_e) * 65535 / (max_e - min_e)), 0, 65535).to(torch.int32)
    q_attr = torch.cat([q_attr_f,q_attr_rest, q_attr_opa, q_attr_scale, q_attr_rot], dim=1)

    space = [[0, 160], [0, 70], [0, 150]]
    colorbook, geobook, codebook = transform.buildVideoTensor(axis, xyz, q_attr, space)
    output_files_col = process.tensor_to_rgb48(colorbook.numpy(), outdir+'/color')
    output_files_geo = process.tensor_to_rgb48(geobook.numpy(), outdir+'/geo')

    if output_files_col is not None and output_files_geo is not None:
        # # 将视频转换回张量
        recovered_tensor_col = process.rgb48_videos_to_tensor(output_files_col, frames_per_video, height, width)
        recovered_tensor_geo = process.rgb48_videos_to_tensor(output_files_geo, frames_per_video, height, width)

        # # 反量化：将 uint8 重新映射回原始范围
        # reconstructed_col = recovered_tensor_col / 65535 * (max_a - min_a) + min_a
        # reconstructed_geo = recovered_tensor_geo / 65535 * (max_a - min_a) + min_a

        # new_attributes = transform.extract_attributes(reconstructed_col, reconstructed_geo, codebook, axis, old_attributes)
        new_attributes = transform.extract_attributes(recovered_tensor_col, recovered_tensor_geo, codebook, axis, old_attributes)
        new_attributes[:, 3:6] = new_attributes[:, 3:6] * (max_a - min_a) / 65535 + min_a
        new_attributes[:, 6:6 + 3 * (3 + 1) ** 2 - 3] = new_attributes[:, 6:6 + 3 * (3 + 1) ** 2 - 3] * (max_b - min_b) / 65535 + min_b
        new_attributes[:, 6 + 3 * (3 + 1) ** 2 - 3:6 + 3 * (3 + 1) ** 2 - 2] = new_attributes[:, 6 + 3 * (3 + 1) ** 2 - 3:6 + 3 * (3 + 1) ** 2 - 2] * (max_c - min_c) / 65535 + min_c
        new_attributes[:, 6 + 3 * (3 + 1) ** 2 - 2:6 + 3 * (3 + 1) ** 2 + 1] = new_attributes[:, 6 + 3 * (3 + 1) ** 2 - 2:6 + 3 * (3 + 1) ** 2 + 1] * (max_d - min_d) / 65535 + min_d
        new_attributes[:, 6 + 3 * (3 + 1) ** 2 + 1:6 + 3 * (3 + 1) ** 2 + 5] = new_attributes[:, 6 + 3 * (3 + 1) ** 2 + 1:6 + 3 * (3 + 1) ** 2 + 5] * (max_e - min_e) / 65535 + min_e

        gs.save_ply(new_attributes, outdir + "/point_cloud_0.ply")

        for i in range(10, 20, 2):
            out = outdir+'/iteration_'+str(i)
            os.makedirs(out, exist_ok=True)
            mpegcomp.process_files(input_dir=outdir+'/color', output_dir=out+'/color', qp=i)
            mpegcomp.process_files(input_dir=outdir+'/geo', output_dir=out+'/geo', qp=i)
            # os.system(' python /data2/zijian/videocomp/main/mpegcomp.py --input_dir /data2/zijian/videocomp/main/output_videos --output_dir '+out+' --width 150 --height 70 --pattern *.rgb48le --pixel_format rgb48le --qp '+str(i))

            new_path_c = update_file_paths(output_files_col,i)
            new_path_g = update_file_paths(output_files_geo,i)

            compressed_tensor_col = process.rgb48_videos_to_tensor(new_path_c, frames_per_video, height, width)
            compressed_tensor_geo = process.rgb48_videos_to_tensor(new_path_g, frames_per_video, height, width)

            compressed_attributes = transform.extract_attributes(compressed_tensor_col,compressed_tensor_geo, codebook, axis, old_attributes)
            compressed_attributes[:, 3:6] = compressed_attributes[:, 3:6] * (max_a - min_a) / 65535 + min_a
            compressed_attributes[:, 6:6 + 3 * (3 + 1) ** 2 - 3] = compressed_attributes[:, 6:6 + 3 * (3 + 1) ** 2 - 3] * (max_b - min_b) / 65535 + min_b
            compressed_attributes[:, 6 + 3 * (3 + 1) ** 2 - 3:6 + 3 * (3 + 1) ** 2 - 2] = compressed_attributes[:, 6 + 3 * (3 + 1) ** 2 - 3:6 + 3 * (3 + 1) ** 2 - 2] * (max_c - min_c) / 65535 + min_c
            compressed_attributes[:, 6 + 3 * (3 + 1) ** 2 - 2:6 + 3 * (3 + 1) ** 2 + 1] = compressed_attributes[:, 6 + 3 * (3 + 1) ** 2 - 2:6 + 3 * (3 + 1) ** 2 + 1] * (max_d - min_d) / 65535 + min_d
            compressed_attributes[:, 6 + 3 * (3 + 1) ** 2 + 1:6 + 3 * (3 + 1) ** 2 + 5] = compressed_attributes[:, 6 + 3 * (3 + 1) ** 2 + 1:6 + 3 * (3 + 1) ** 2 + 5] * (max_e - min_e) / 65535 + min_e
            ex_attr2 = compressed_attributes[:, 6:15]
            gs.save_ply(compressed_attributes, out + "/point_cloud.ply")


    # print(x_reconstructed)
