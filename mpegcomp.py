import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor
import glob


def yuv_to_video(input_yuv, output_video, width, height, framerate, pixel_format='yuv420p'):
    """
    将YUV文件转换为视频文件
    """
    cmd = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-pixel_format', pixel_format,
        '-video_size', f'{width}x{height}',
        '-framerate', str(framerate),
        '-i', input_yuv,
        '-c:v', 'libx264',
        '-y',
        output_video
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"成功将 {input_yuv} 转换为 {output_video}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"转换失败 {input_yuv}: {e}")
        return False


def video_to_yuv(input_video, output_yuv, pixel_format='yuv420p'):
    """
    将视频文件转换回YUV格式
    """
    cmd = [
        'ffmpeg',
        '-i', input_video,
        '-c:v', 'rawvideo',
        '-pixel_format', pixel_format,
        '-y',
        output_yuv
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"成功将 {input_video} 转换为 {output_yuv}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"转换失败 {input_video}: {e}")
        return False


def process_single_file(input_yuv, output_dir, width, height, framerate, pixel_format):
    """
    处理单个YUV文件的完整流程
    """
    base_name = os.path.splitext(os.path.basename(input_yuv))[0]
    temp_video = os.path.join(output_dir, f"{base_name}_temp.mp4")
    final_yuv = os.path.join(output_dir, f"{base_name}.yuv")

    if yuv_to_video(input_yuv, temp_video, width, height, framerate, pixel_format):
        if video_to_yuv(temp_video, final_yuv, pixel_format):
            # 清理临时视频文件
            # os.remove(temp_video)
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description='批量处理YUV文件：YUV -> 视频 -> YUV')
    parser.add_argument('--input_dir', help='输入目录路径（包含YUV文件）')
    parser.add_argument('--output_dir', help='输出目录路径')
    parser.add_argument('--width', type=int, required=True, help='视频宽度')
    parser.add_argument('--height', type=int, required=True, help='视频高度')
    parser.add_argument('--framerate', type=int, default=30, help='帧率（默认30）')
    parser.add_argument('--pixel_format', default='yuv420p', help='像素格式（默认yuv420p）')
    parser.add_argument('--pattern', default='*.yuv', help='YUV文件匹配模式（默认*.yuv）')
    parser.add_argument('--max_workers', type=int, default=4, help='最大并行处理数（默认4）')

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取所有匹配的YUV文件
    yuv_files = glob.glob(os.path.join(args.input_dir, args.pattern))

    if not yuv_files:
        print(f"在 {args.input_dir} 中没有找到匹配 {args.pattern} 的YUV文件")
        return

    print(f"找到 {len(yuv_files)} 个YUV文件待处理")

    # 使用线程池并行处理文件
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                process_single_file,
                input_yuv,
                args.output_dir,
                args.width,
                args.height,
                args.framerate,
                args.pixel_format
            )
            for input_yuv in yuv_files
        ]

        # 等待所有任务完成
        completed = 0
        for future in futures:
            if future.result():
                completed += 1

    print(f"处理完成！成功转换 {completed}/{len(yuv_files)} 个文件")


if __name__ == '__main__':
    main()