#
# import os
# import subprocess
# import glob
# import re
# from concurrent.futures import ThreadPoolExecutor
# from typing import Dict, Optional, Callable, Union
#
#
# def raw_to_video(input_path, output_video, width, height, framerate, input_pixel_format, output_pixel_format, qp):
#     """
#     将原始视频文件转换为编码视频文件
#
#     Args:
#         input_path (str): 输入文件路径
#         output_video (str): 输出视频文件路径
#         width (int): 视频宽度
#         height (int): 视频高度
#         framerate (int): 帧率
#         input_pixel_format (str): 输入像素格式
#         output_pixel_format (str): 输出像素格式
#         qp (int): 量化参数
#     """
#     cmd = [
#         'ffmpeg',
#         '-f', 'rawvideo',
#         '-pix_fmt', input_pixel_format,
#         '-video_size', f'{width}x{height}',
#         '-framerate', str(framerate),
#         '-i', input_path,
#         '-pix_fmt', output_pixel_format,
#         '-c:v', 'libx265',
#         '-qp', str(qp),
#         # '-x265-params', 'keyint=999999:min-keyint=999999:scenecut=0',
#         # '-keyint_min', '200',
#         # '-g', '200',
#         '-y',
#         output_video
#     ]
#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         return True
#     except subprocess.CalledProcessError as e:
#         print(f"视频编码失败 {input_path}: {e}")
#         return False
#
#
# def video_to_raw(input_video, output_path, output_pixel_format):
#     """
#     将编码视频文件转换回原始格式
#
#     Args:
#         input_video (str): 输入视频文件路径
#         output_path (str): 输出文件路径
#         output_pixel_format (str): 输出像素格式
#     """
#     cmd = [
#         'ffmpeg',
#         '-i', input_video,
#         '-f', 'rawvideo',
#         '-pix_fmt', output_pixel_format,
#         '-y',
#         output_path
#     ]
#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         return True
#     except subprocess.CalledProcessError as e:
#         print(f"解码失败 {input_video}: {e}")
#         return False
#
#
# def get_intermediate_format(filename: str, default_format: str = 'gbrp12le') -> str:
#     """
#     根据文件名决定使用的中间像素格式
#
#     Args:
#         filename (str): 文件名
#         default_format (str): 默认像素格式
#
#     Returns:
#         str: 选定的像素格式
#     """
#     # 使用正则表达式匹配color_1到color_15的模式
#     if re.match(r'color_(?:[1-9]|1[0-5])', os.path.splitext(filename)[0]):
#         return 'yuv444p12le'
#     return default_format
#
# def get_qp(filename: str, default_qp) -> int:
#
#     # 使用正则表达式匹配color_1到color_15的模式
#     if re.match(r'color_(?:[1-9]|1[0-5])', os.path.splitext(filename)[0]):
#         return default_qp
#     return 0
#
#
# def process_single_file(
#         input_file: str,
#         output_dir: str,
#         width: int,
#         height: int,
#         framerate: int,
#         input_pixel_format: str,
#         output_pixel_format: str,
#         d_qp: int,
#         default_intermediate_format: str = 'gbrp12le'
# ) -> bool:
#     """
#     处理单个文件的完整流程
#
#     Args:
#         input_file (str): 输入文件路径
#         output_dir (str): 输出目录路径
#         width (int): 视频宽度
#         height (int): 视频高度
#         framerate (int): 帧率
#         input_pixel_format (str): 输入像素格式
#         output_pixel_format (str): 输出像素格式
#         qp (int): 量化参数
#         default_intermediate_format (str): 默认中间像素格式
#
#     Returns:
#         bool: 处理是否成功
#     """
#     base_name = os.path.splitext(os.path.basename(input_file))[0]
#     temp_video = os.path.join(output_dir, f"{base_name}_temp.hevc")
#     final_output = os.path.join(output_dir, f"{base_name}{os.path.splitext(input_file)[1]}")
#
#     # 确定中间像素格式
#     intermediate_format = get_intermediate_format(
#         os.path.basename(input_file),
#         default_intermediate_format
#     )
#
#     t_qp = get_qp(
#         os.path.basename(input_file),
#         d_qp
#     )
#
#     if raw_to_video(input_file, temp_video, width, height, framerate,
#                     input_pixel_format, intermediate_format, t_qp):
#         if video_to_raw(temp_video, final_output, output_pixel_format):
#             # 可选：删除临时视频文件
#             # os.remove(temp_video)
#             return True
#     return False
#
#
# def process_files(
#         input_dir: str,
#         output_dir: str,
#         width: int = 150,
#         height: int = 70,
#         framerate: int = 160,
#         input_pixel_format: str = 'rgb48le',
#         output_pixel_format: str = 'rgb48le',
#         pattern: str = '*.rgb48le',
#         max_workers: int = 4,
#         qp: int = 30,
#         default_intermediate_format: str = 'gbrp12le'
# ) -> None:
#     """
#     批量处理文件
#
#     Args:
#         input_dir (str): 输入目录
#         output_dir (str): 输出目录
#         width (int): 视频宽度
#         height (int): 视频高度
#         framerate (int): 帧率
#         input_pixel_format (str): 输入像素格式
#         output_pixel_format (str): 输出像素格式
#         pattern (str): 文件匹配模式
#         max_workers (int): 最大并行处理数
#         qp (int): 量化参数
#         default_intermediate_format (str): 默认中间像素格式
#     """
#     os.makedirs(output_dir, exist_ok=True)
#
#     input_files = glob.glob(os.path.join(input_dir, pattern))
#     if not input_files:
#         print(f"在 {input_dir} 中没有找到匹配 {pattern} 的文件")
#         return
#
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [
#             executor.submit(
#                 process_single_file,
#                 input_file,
#                 output_dir,
#                 width,
#                 height,
#                 framerate,
#                 input_pixel_format,
#                 output_pixel_format,
#                 qp,
#                 default_intermediate_format
#             )
#             for input_file in input_files
#         ]
#
#         completed = 0
#         for future in futures:
#             if future.result():
#                 completed += 1
#
#     print(f"处理完成！成功转换 {completed}/{len(input_files)} 个文件")

import os
import subprocess
import glob
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Callable, Union


def raw_to_video(input_path, output_video, width, height, framerate, input_pixel_format, output_pixel_format, qp=None):
    """
    Convert raw video file to encoded video file with flexible encoding parameters

    Args:
        input_path (str): Input file path
        output_video (str): Output video file path
        width (int): Video width
        height (int): Video height
        framerate (int): Frame rate
        input_pixel_format (str): Input pixel format
        output_pixel_format (str): Output pixel format
        crf (int): Constant Rate Factor for lossless/near-lossless encoding
        qp (int): Quantization Parameter
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
    ]

    # Add encoding parameters based on CRF or QP
    if qp is None:
        cmd.extend(['-x265-params', 'lossless=1'])
        # cmd.extend(['-x265-params',"bframes=0:keyint=1",'-qp', '0'])
    elif qp > 0:
        cmd.extend(['-qp', str(qp)])
    else:
        # Default to lossless encoding
        cmd.extend(['-qp', '0'])

    cmd.extend([
        '-y',
        output_video
    ])

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Video encoding failed for {input_path}: {e}")
        return False


def video_to_raw(input_video, output_path, output_pixel_format):
    """
    Convert encoded video file back to raw format

    Args:
        input_video (str): Input video file path
        output_path (str): Output file path
        output_pixel_format (str): Output pixel format
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
        print(f"Decoding failed for {input_video}: {e}")
        return False


def process_directory(
        input_dir: str,
        output_dir: str,
        width: int = 100,
        height: int = 100,
        framerate: int = 60,
        input_pixel_format: str = 'rgb48le',
        max_workers: int = 4,
        qp: int = None
) -> None:
    """
    Process files in different subdirectories with specialized encoding

    Args:
        input_dir (str): Root input directory
        output_dir (str): Root output directory
        width (int): Video width
        height (int): Video height
        framerate (int): Frame rate
        input_pixel_format (str): Input pixel format
        output_pixel_format (str): Output pixel format
        max_workers (int): Maximum parallel processing workers
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_pixel_format ={'y':'yuv444p12le','g':'gbrp12le'}
    # Process files in each subdirectory
    subdirs = ['coord', 'color', 'geo']

    for subdir in subdirs:
        input_subdir = os.path.join(input_dir, subdir)
        output_subdir = os.path.join(output_dir, subdir)
        os.makedirs(output_subdir, exist_ok=True)

        # Find all RGB files
        input_files = sorted(glob.glob(os.path.join(input_subdir, '*.rgb48le')))

        if not input_files:
            print(f"No files found in {input_subdir}")
            continue

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for i, input_file in enumerate(input_files):
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                temp_video = os.path.join(output_subdir, f"{base_name}_temp.hevc")
                final_output = os.path.join(output_subdir, f"{base_name}{os.path.splitext(input_file)[1]}")

                # Encoding strategy based on subdirectory
                if subdir == 'coord':
                    # Lossless encoding for coord files
                    futures.append(
                        executor.submit(
                            process_single_file,
                            input_file, temp_video, final_output,
                            width, height, framerate,
                            input_pixel_format, output_pixel_format['g'],
                            qp=None  # Lossless encoding for coord files
                        )
                    )
                elif subdir == 'color':
                    # First file CRF 10, others with increased QP
                    if i == 0:
                        futures.append(
                            executor.submit(
                                process_single_file,
                                input_file, temp_video, final_output,
                                width, height, framerate,
                                input_pixel_format, output_pixel_format['g'],
                                qp=0
                            )
                        )
                    else:
                        futures.append(
                            executor.submit(
                                process_single_file,
                                input_file, temp_video, final_output,
                                width, height, framerate,
                                input_pixel_format, output_pixel_format['y'],
                                qp=qp  # Increased QP for other files
                            )
                        )
                elif subdir == 'geo':

                    futures.append(
                        executor.submit(
                            process_single_file,
                            input_file, temp_video, final_output,
                            width, height, framerate,
                            input_pixel_format, output_pixel_format['g'],
                            qp=0
                        )
                    )

            # Wait for all futures to complete and track results
            completed = sum(1 for future in futures if future.result())
            print(f"Processed {subdir}: {completed}/{len(input_files)} files successfully")


def process_single_file(
        input_file: str,
        temp_video: str,
        final_output: str,
        width: int,
        height: int,
        framerate: int,
        input_pixel_format: str,
        mid_pixel_format: str,
        qp: int = None
) -> bool:
    """
    Process a single file with specialized encoding

    Args:
        input_file (str): Input file path
        temp_video (str): Temporary video file path
        final_output (str): Final output file path
        width (int): Video width
        height (int): Video height
        framerate (int): Frame rate
        input_pixel_format (str): Input pixel format
        output_pixel_format (str): Output pixel format

    Returns:
        bool: Processing success status
    """
    if raw_to_video(input_file, temp_video, width, height, framerate,
                    input_pixel_format, mid_pixel_format, qp=qp):
        if video_to_raw(temp_video, final_output, 'rgb48le'):
            # Optional: Remove temporary video file
            # os.remove(temp_video)
            return True
    return False


def main(input_dir: str, output_dir: str):
    """
    Main function to process entire directory structure

    Args:
        input_dir (str): Root input directory
        output_dir (str): Root output directory
    """
    process_directory(input_dir, output_dir)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_directory>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])