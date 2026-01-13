import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

def download_coco128(root_dir='datasets'):
    """下载并解压 COCO128 数据集"""
    # 1. 设置路径
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    
    url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip'
    zip_path = root / 'coco128.zip'
    extract_path = root / 'coco128'

    # 2. 检查是否已经存在
    if extract_path.exists():
        print(f"✅ 数据集已存在于: {extract_path}")
        return

    # 3. 下载
    print(f"⬇️ 正在下载 COCO128 到 {root} ...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

    # 4. 解压
    print("📦 正在解压...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root)
    
    # 5. 清理压缩包
    os.remove(zip_path)
    print(f"🎉 完成！数据集准备就绪: {root / 'coco128'}")

if __name__ == "__main__":
    # 需要安装 requests 和 tqdm: pip install requests tqdm
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("正在安装依赖库...")
        os.system("pip install requests tqdm")
        import requests
        from tqdm import tqdm
        
    download_coco128()
