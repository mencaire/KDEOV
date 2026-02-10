import torch
import sys
import os

# 这一步是为了让 Python 能找到 models 文件夹里的代码
sys.path.append(os.getcwd())

try:
    print("Step 1: 尝试导入 LightweightVisualBackbone...")
    from models.components import LightweightVisualBackbone
    print("✅ 导入成功！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确认你的 test_backbone.py 是在项目的根目录下，并且 models 文件夹里有 __init__.py")
    exit()

def test_yolo():
    print("\nStep 2: 正在初始化 YOLOv8n 模型 (权重从 weights/ 加载或下载)...")
    try:
        # 实例化模型，yolov8n.pt 从 weights/ 加载或下载到 weights/
        backbone = LightweightVisualBackbone(backbone_type="yolov8n", pretrained=True, weights_dir="weights")
        print("✅ 模型初始化成功！")
    except Exception as e:
        print(f"❌ 模型初始化崩溃: {e}")
        return

    print("\nStep 3: 构造一张假图片进行测试...")
    # 构造一个 Batch=1, Channel=3, Height=640, Width=640 的随机张量
    fake_image = torch.randn(1, 3, 640, 640)
    
    print("Step 4: 开始前向传播 (Forward Pass)...")
    try:
        # 这一步最容易报错
        features = backbone(fake_image)
        
        # 检查输出
        if isinstance(features, (list, tuple)):
            print(f"✅ 运行成功！输出了 {len(features)} 层特征。")
            for i, f in enumerate(features):
                print(f"   -> 第 {i+1} 层特征形状: {f.shape}")
        else:
            print(f"✅ 运行成功！输出形状: {features.shape}")
            
    except Exception as e:
        print("\n" + "="*40)
        print("❌ 这里的代码有问题！(这就是我们要找的 Bug)")
        print(f"错误信息: {e}")
        print("="*40)
        print("原因分析: 简单的 model[:10] 切片破坏了 YOLO 的内部连接结构。")

if __name__ == "__main__":
    test_yolo()
