import os
import zipfile
import requests
import shutil
from tqdm import tqdm
import argparse

# ----------------------
# 配置：数据集路径
# ----------------------
DATA_ROOT = "datasets"
COCO128_DIR = os.path.join(DATA_ROOT, "coco128")
COCO2017_DIR = os.path.join(DATA_ROOT, "coco2017")

# COCO 2017 下载链接
COCO2017_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

# ----------------------
# 工具函数
# ----------------------
def download_file(url, dest_path):
    """下载文件并显示进度条"""
    if os.path.exists(dest_path):
        print(f"⚠️ 文件已存在，跳过下载: {dest_path}")
        return
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"📥 开始下载: {os.path.basename(dest_path)}...")
    
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024  # 优化：改为 1MB，提高写入效率
        
        with open(dest_path, "wb") as f, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
        print(f"✅ 下载完成: {dest_path}")
    except KeyboardInterrupt:
        print("\n🛑 下载被中断，正在删除未完成的文件...")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)

def extract_zip(zip_path, extract_to):
    """解压 zip 文件"""
    if not os.path.exists(zip_path):
        print(f"❌ 错误: 找不到文件 {zip_path}")
        return
    
    print(f"📦 正在解压: {os.path.basename(zip_path)} -> {extract_to} ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ 解压完成")
    except zipfile.BadZipFile:
        print(f"❌ 错误: Zip 文件损坏 {zip_path}")

# ----------------------
# 功能模块
# ----------------------
def download_coco128():
    """下载 COCO128 (128张图，用于测试代码)"""
    if os.path.exists(COCO128_DIR):
        print(f"✅ COCO128 已存在: {COCO128_DIR}")
        return
    
    print("🚀 正在获取 COCO128 (迷你数据集)...")
    url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    zip_path = os.path.join(DATA_ROOT, "coco128.zip")
    
    download_file(url, zip_path)
    extract_zip(zip_path, DATA_ROOT)
    
    # 清理压缩包
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("🗑️ 已删除压缩包以节省空间")

def download_coco2017(parts):
    """下载 COCO 2017 (完整数据集)"""
    os.makedirs(COCO2017_DIR, exist_ok=True)
    
    for part in parts:
        if part not in COCO2017_URLS:
            continue
            
        url = COCO2017_URLS[part]
        # 注意：annotations 的 zip 文件名比较长，我们简化一下
        filename = f"{part}.zip" 
        zip_path = os.path.join(COCO2017_DIR, filename)
        
        # 1. 检查文件夹是否已存在（避免重复下载）
        # annotations 解压后是 annotations 文件夹
        # train2017 解压后是 train2017 文件夹
        expected_dir_name = "annotations" if part == "annotations" else part
        if os.path.exists(os.path.join(COCO2017_DIR, expected_dir_name)):
            print(f"✅ {expected_dir_name} 已存在，跳过。")
            continue

        # 2. 下载
        download_file(url, zip_path)
        
        # 3. 解压
        extract_zip(zip_path, COCO2017_DIR)
        
        # 4. 删除压缩包 (关键步骤：节省空间！)
        if os.path.exists(zip_path):
            print(f"🗑️ 删除压缩包: {filename} (释放空间)")
            os.remove(zip_path)

# ----------------------
# 主程序
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KDEOV 数据集下载助手")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="coco128", 
        choices=["coco128", "coco2017"],
        help="选择数据集: coco128 (测试用) 或 coco2017 (训练用)"
    )
    parser.add_argument(
        "--parts", 
        nargs="+", 
        default=["train2017", "val2017", "annotations"],
        help="COCO2017 下载部分: train2017 val2017 annotations"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "coco128":
        download_coco128()
    elif args.dataset == "coco2017":
        download_coco2017(parts=args.parts)
        
    print("\n🎉 所有任务完成！")

