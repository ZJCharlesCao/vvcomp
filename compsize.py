import os

def get_hevc_files_size(directory: str) -> int:
    total_size = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.hevc'):
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
    return total_size

def calculate_hevc_sizes(base_path: str):
    results = {}
    for i in range(0, 2):
        dir_name = f"iteration_{i}"
        dir_path = os.path.join(base_path, dir_name)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            total_size = get_hevc_files_size(dir_path)
            results[dir_name] = total_size
    return results

# Example usage
base_path = '/data2/zijian/videocomp/main/output_videos'
hevc_sizes = calculate_hevc_sizes(base_path)
for dir_name, size in hevc_sizes.items():
    print(f"Total size of HEVC files in {dir_name}: {size} bytes")