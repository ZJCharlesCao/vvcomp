# import os
# import subprocess
# import argparse
# from concurrent.futures import ThreadPoolExecutor
# import glob
#
#
# def raw_to_video(input, output_video, width, height, framerate, pixel_format,qp):
#     """
#     将YUV文件转换为视频文件
#     """
#     cmd = [
#         'ffmpeg',
#         '-f', 'rawvideo',
#         '-pix_fmt', pixel_format,
#         '-video_size', f'{width}x{height}',
#         '-framerate', str(framerate),
#         '-i', input,
#         # '-pix_fmt', 'yuv444p12le',
#         '-c:v', 'libx265',
#         # '-x265-params', 'lossless=1',
#         '-qp', str(qp),
#         '-y',
#         output_video
#     ]
#
#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         print(f"成功将 {input} 转换为 {output_video}")
#         return True
#     except subprocess.CalledProcessError as e:
#         print(f"转换失败 {input}: {e}")
#         return False
#
#
# def video_to_raw(input_video, output, pixel_format):
#     """
#     将视频文件转换回YUV格式
#     """
#     cmd = [
#         'ffmpeg',
#         '-i', input_video,
#         '-f', 'rawvideo',
#         '-pix_fmt', pixel_format,
#         '-y',
#         output
#     ]
#
#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         print(f"成功将 {input_video} 转换为 {output}")
#         return True
#     except subprocess.CalledProcessError as e:
#         print(f"转换失败 {input_video}: {e}")
#         return False
#
#
# def process_single_file(input_yuv, output_dir, width, height, framerate, pixel_format,qp):
#     """
#     处理单个YUV文件的完整流程
#     """
#     base_name = os.path.splitext(os.path.basename(input_yuv))[0]
#     temp_video = os.path.join(output_dir, f"{base_name}_temp.hevc")
#     final_yuv = os.path.join(output_dir, f"{base_name}.rgb48le")
#
#     if raw_to_video(input_yuv, temp_video, width, height, framerate, pixel_format,qp):
#         if video_to_raw(temp_video, final_yuv, pixel_format):
#             # 清理临时视频文件
#             # os.remove(temp_video)
#             return True
#     return False
#
#
# def main():
#     parser = argparse.ArgumentParser(description='批量处理YUV文件：YUV -> 视频 -> YUV')
#     parser.add_argument('--input_dir', help='输入目录路径（包含YUV文件）')
#     parser.add_argument('--output_dir', help='输出目录路径')
#     parser.add_argument('--width', type=int, required=True, help='视频宽度')
#     parser.add_argument('--height', type=int, required=True, help='视频高度')
#     parser.add_argument('--framerate', type=int, default=30, help='帧率（默认30）')
#     parser.add_argument('--pixel_format', default='yuv420p', help='像素格式（默认yuv420p）')
#     parser.add_argument('--pattern', default='*.yuv', help='YUV文件匹配模式（默认*.yuv）')
#     parser.add_argument('--max_workers', type=int, default=4, help='最大并行处理数（默认4）')
#     parser.add_argument('--qp', type=int,help='qp值')
#
#     args = parser.parse_args()
#
#     # 确保输出目录存在
#     os.makedirs(args.output_dir, exist_ok=True)
#
#     # 获取所有匹配的YUV文件
#     yuv_files = glob.glob(os.path.join(args.input_dir, args.pattern))
#
#     if not yuv_files:
#         print(f"在 {args.input_dir} 中没有找到匹配 {args.pattern} 的raw文件")
#         return
#
#     print(f"找到 {len(yuv_files)} 个YUV文件待处理")
#
#     # 使用线程池并行处理文件
#     with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
#         futures = [
#             executor.submit(
#                 process_single_file,
#                 input_yuv,
#                 args.output_dir,
#                 args.width,
#                 args.height,
#                 args.framerate,
#                 args.pixel_format,
#                 args.qp
#             )
#             for input_yuv in yuv_files
#         ]
#
#         # 等待所有任务完成
#         completed = 0
#         for future in futures:
#             if future.result():
#                 completed += 1
#
#     print(f"处理完成！成功转换 {completed}/{len(yuv_files)} 个文件")
#
#
# if __name__ == '__main__':
#     main()


import os
import subprocess
import glob
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Callable, Union


def raw_to_video(input_path, output_video, width, height, framerate, input_pixel_format, output_pixel_format, qp):
    """
    将原始视频文件转换为编码视频文件

    Args:
        input_path (str): 输入文件路径
        output_video (str): 输出视频文件路径
        width (int): 视频宽度
        height (int): 视频高度
        framerate (int): 帧率
        input_pixel_format (str): 输入像素格式
        output_pixel_format (str): 输出像素格式
        qp (int): 量化参数
    """
    cmd = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-pix_fmt', input_pixel_format,
        '-video_size', f'{width}x{height}',
        '-framerate', str(framerate),
        '-i', input_path,
        '-pix_fmt', output_pixel_format,
        '-c:v', 'libx265',
        '-qp', str(qp),
        '-y',
        output_video
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"视频编码失败 {input_path}: {e}")
        return False


def video_to_raw(input_video, output_path, output_pixel_format):
    """
    将编码视频文件转换回原始格式

    Args:
        input_video (str): 输入视频文件路径
        output_path (str): 输出文件路径
        output_pixel_format (str): 输出像素格式
    """
    cmd = [
        'ffmpeg',
        '-i', input_video,
        '-f', 'rawvideo',
        '-pix_fmt', output_pixel_format,
        '-y',
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"解码失败 {input_video}: {e}")
        return False


def get_intermediate_format(filename: str, default_format: str = 'gbrp12le') -> str:
    """
    根据文件名决定使用的中间像素格式

    Args:
        filename (str): 文件名
        default_format (str): 默认像素格式

    Returns:
        str: 选定的像素格式
    """
    # 使用正则表达式匹配color_1到color_15的模式
    if re.match(r'color_(?:[1-9]|1[0-5])', os.path.splitext(filename)[0]):
        return 'yuv444p12le'
    return default_format

def get_qp(filename: str, default_qp) -> int:

    # 使用正则表达式匹配color_1到color_15的模式
    if re.match(r'color_(?:[1-9]|1[0-5])', os.path.splitext(filename)[0]):
        return default_qp
    return 0


def process_single_file(
        input_file: str,
        output_dir: str,
        width: int,
        height: int,
        framerate: int,
        input_pixel_format: str,
        output_pixel_format: str,
        d_qp: int,
        default_intermediate_format: str = 'gbrp12le'
) -> bool:
    """
    处理单个文件的完整流程

    Args:
        input_file (str): 输入文件路径
        output_dir (str): 输出目录路径
        width (int): 视频宽度
        height (int): 视频高度
        framerate (int): 帧率
        input_pixel_format (str): 输入像素格式
        output_pixel_format (str): 输出像素格式
        qp (int): 量化参数
        default_intermediate_format (str): 默认中间像素格式

    Returns:
        bool: 处理是否成功
    """
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    temp_video = os.path.join(output_dir, f"{base_name}_temp.hevc")
    final_output = os.path.join(output_dir, f"{base_name}{os.path.splitext(input_file)[1]}")

    # 确定中间像素格式
    intermediate_format = get_intermediate_format(
        os.path.basename(input_file),
        default_intermediate_format
    )

    t_qp = get_qp(
        os.path.basename(input_file),
        d_qp
    )

    if raw_to_video(input_file, temp_video, width, height, framerate,
                    input_pixel_format, intermediate_format, t_qp):
        if video_to_raw(temp_video, final_output, output_pixel_format):
            # 可选：删除临时视频文件
            # os.remove(temp_video)
            return True
    return False


def process_files(
        input_dir: str,
        output_dir: str,
        width: int = 150,
        height: int = 70,
        framerate: int = 160,
        input_pixel_format: str = 'rgb48le',
        output_pixel_format: str = 'rgb48le',
        pattern: str = '*.rgb48le',
        max_workers: int = 4,
        qp: int = 30,
        default_intermediate_format: str = 'gbrp12le'
) -> None:
    """
    批量处理文件

    Args:
        input_dir (str): 输入目录
        output_dir (str): 输出目录
        width (int): 视频宽度
        height (int): 视频高度
        framerate (int): 帧率
        input_pixel_format (str): 输入像素格式
        output_pixel_format (str): 输出像素格式
        pattern (str): 文件匹配模式
        max_workers (int): 最大并行处理数
        qp (int): 量化参数
        default_intermediate_format (str): 默认中间像素格式
    """
    os.makedirs(output_dir, exist_ok=True)

    input_files = glob.glob(os.path.join(input_dir, pattern))
    if not input_files:
        print(f"在 {input_dir} 中没有找到匹配 {pattern} 的文件")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_file,
                input_file,
                output_dir,
                width,
                height,
                framerate,
                input_pixel_format,
                output_pixel_format,
                qp,
                default_intermediate_format
            )
            for input_file in input_files
        ]

        completed = 0
        for future in futures:
            if future.result():
                completed += 1

    print(f"处理完成！成功转换 {completed}/{len(input_files)} 个文件")
