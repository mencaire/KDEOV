# Development Log

## Implementation Phase

### 2026/02/04 Peng - 开放词汇目标检测实现与文档更新

#### 主要工作内容

1. **开放词汇目标检测（Open-Vocabulary Object Detection）实现**
   - 在 `models/components.py` 中新增 **SpatialProjection**：1×1 卷积 + GroupNorm，将 backbone 特征图映射到 CLIP 嵌入空间并**保留空间维度**（不做全局池化），输出形状 `[B, 512, Hf, Wf]`，用于每个位置与文本的相似度计算
   - 新增 **grid_boxes_to_image** 工具函数：将特征图网格映射到图像坐标的默认框（xyxy 格式），供检测头使用
   - 在 `models/kdeov_model.py` 中新增 **get_spatial_embeddings()**：获取每位置图像嵌入（可选与文本融合）
   - 新增 **open_vocabulary_detect()**：输入图像和类别名称列表，输出每张图的检测结果（boxes、scores、labels）；流程为：编码类别文本 → 空间嵌入 → 与类别嵌入相似度 → 网格默认框 → 阈值 + NMS（使用 torchvision.ops.nms）→ 返回检测列表
   - 模型形成**双路径**：分类/检索路径使用 Projection Network（全局池化）；检测路径使用 Spatial Projection（保留空间），共享 backbone 与 Fusion Module

2. **model_summary.py 更新**
   - **analyze_model_components**：增加 spatial_projection 组件的统计与描述
   - **架构图**：更新为双路径结构（Projection Network 与 Spatial Projection 分支），并标注分类/检索与开放词汇检测两条输出
   - **test_forward_pass**：增加 get_spatial_embeddings 与 open_vocabulary_detect 的 shape 测试（检测部分用 try-except 包裹，避免缺少 torchvision 时报错）
   - **print_detailed_summary**：输出中增加空间嵌入与检测结果的 shape；训练组件列表中标明 Projection Network（分类/检索路径）与 Spatial Projection（检测路径）
   - **print_static_summary**：增加第 5 组件 SpatialProjection 的说明及“两条视觉输出路径”描述，更新参数与体积估算；使用示例中增加 open_vocabulary_detect 调用

3. **models/README.md 更新**
   - 开头增加**当前模型结构概要**：文本流、视觉流及两条路径（分类/检索 vs 检测）
   - **Model Components** 增加 **§5. Spatial Projection**：作用、输入输出形状、与 Projection Network 的区分
   - **Architecture Details** 增加 **Model Structure Overview** 表格：两条路径的用途与输出
   - **Inference Flow** 拆分为：图像级（Projection Network → 图像嵌入 → 相似度）与开放词汇检测（Spatial Projection → 每位置嵌入 → 与类别文本相似度 + 网格框 + NMS）

4. **主项目 README.md 更新**
   - **Project Overview** 下增加 **Current Model Structure (High Level)**，简述双路径并链接至 models/README.md
   - **Current Progress - Model Implementation**：补充 Spatial Projection、双路径描述、open_vocabulary_detect / get_spatial_embeddings
   - **Core Model Files**：`__init__.py` 导出 SpatialProjection、grid_boxes_to_image；components.py、kdeov_model.py 的功能说明与“关键功能”列表更新为包含检测
   - **model_summary.py** 描述更新：架构图、组件分析、输入输出形状、输出章节均体现双路径与检测
   - **Usage Guide** 新增 **§7. Open-Vocabulary Object Detection**：完整示例（open_vocabulary_detect 参数与结果解析）
   - **Model Execution Workflow**：推理步骤中明确列出零样本分类、检索与开放词汇检测

5. **包导出**
   - 在 `models/__init__.py` 中导出 `SpatialProjection` 与 `grid_boxes_to_image`

#### 技术细节

- **SpatialProjection**：Conv2d(1×1) + GroupNorm，再在通道维做 L2 归一化，输出 `[B, D, Hf, Wf]`
- **默认框**：每个特征网格单元对应一个框，中心为 `(j+0.5)*stride_x`, `(i+0.5)*stride_y`，尺寸由 `cell_scale` 控制
- **检测后处理**：先按 score_threshold 过滤，再 class-agnostic NMS；若无 torchvision.ops.nms 则仅做 top-k 截断

#### 成果

- ✅ 模型支持开放词汇目标检测，可通过 `open_vocabulary_detect(images, class_names, ...)` 得到 boxes、scores、labels
- ✅ 模型结构文档（models/README.md、主 README.md）与当前双路径设计一致
- ✅ model_summary.py 可展示双路径架构及检测相关 shape，静态摘要含 Spatial Projection 与检测用法

---

### 2026/01/19 Peng - 模型结构检查和可视化工具开发

#### 主要工作内容

1. **模型结构全面检查**
   - 对 `models/` 目录下的所有模型代码进行了全面审查
   - 识别并修复了多个关键问题：
     - **YOLO Hook 注册问题**：修复了 YOLO backbone 中 hooks 无法捕获特征的问题，改为递归注册到所有 Conv2d 层
     - **Dtype 不匹配问题**：修复了 CLIP 文本编码器返回 float16 而其他组件期望 float32 的问题，在 `FrozenCLIPTextEncoder` 中统一转换为 float32
     - **YOLO 参数可训练性**：确保 YOLO 模型参数正确设置为可训练状态（`requires_grad=True`）
     - **错误处理改进**：增强了 YOLO 初始化和 forward pass 的错误处理机制，提供更好的回退策略

2. **模型可视化工具开发**
   - 创建了 `model_summary.py` 脚本，提供全面的模型分析功能：
     - **模型架构可视化**：文本形式的架构图，展示数据流和组件关系
     - **参数统计**：自动统计总参数、可训练参数、冻结参数，并格式化显示（K/M/B）
     - **组件详细分析**：逐个组件显示参数数量和模型大小
     - **输入/输出形状测试**：自动测试 forward pass 并捕获所有张量形状
     - **内存使用估算**：计算模型大小（MB）
     - **训练信息**：列出可训练组件和损失函数
     - **模型对比**：与完整 CLIP 模型进行大小对比
     - **静态摘要模式**：即使没有安装 CLIP 也能显示架构概览
   - 支持命令行参数：`--backbone`, `--fusion`, `--static`
   - 处理了 Windows 控制台编码问题，确保在不同平台上正常显示

3. **文档更新**
   - 在 `README.md` 中添加了 `model_summary.py` 的详细说明：
     - 功能描述和使用方法
     - 命令行参数说明
     - 使用场景和输出内容
     - 在 Quick Start 部分添加了模型结构验证步骤
   - 更新了 Repository Structure 和 File Function Quick Reference

4. **验证和测试**
   - 验证了所有修复后的功能正常工作
   - 确认 YOLO hooks 成功捕获多尺度特征
   - 确认 forward pass 测试通过，无 dtype 错误
   - 确认参数统计准确（可训练参数从 156K 正确显示为 3.31M）

#### 技术细节

- **YOLO Hook 改进**：从简单的顶层注册改为递归注册到所有 Conv2d 层，深度限制为 15 层
- **Dtype 处理**：在 `FrozenCLIPTextEncoder.forward()` 中添加了 dtype 检查和转换逻辑
- **参数可训练性**：在 YOLO 模型加载后显式设置所有参数为 `requires_grad=True`
- **错误处理**：添加了多层 try-except 块，确保在 YOLO 初始化失败时能优雅地回退到 simple CNN backbone

#### 成果

- ✅ 模型结构检查完成，所有关键问题已修复
- ✅ 创建了功能完善的模型可视化工具
- ✅ 文档更新完成，用户可以通过 `model_summary.py` 快速了解模型结构
- ✅ 模型已准备好用于训练和进一步开发

### 2025/12/23 Peng - 模型运行验证和设备兼容性改进

#### 主要工作内容

1. **模型运行验证**
   - 确认 models 模型文件基本可运行
   - 验证了 KDEOVModel 及其组件的正确性
   - 测试了模型前向传播和损失计算功能

2. **训练脚本优化**
   - `train_feature_alignment.py` 可正常运行
   - 改进了 loss 可视化功能，使用 matplotlib 绘制训练曲线
   - 添加了实时 loss 变化趋势显示和收敛分析
   - 增强了训练过程的输出信息，包括损失变化、收敛状态等

3. **设备兼容性改进**
   - 为了支持多设备（Mac MPS、NVIDIA CUDA、CPU），将 `.cuda()` 改回 `.to(device)`
   - 添加了自动设备检测功能，支持 Mac MPS、NVIDIA CUDA 和 CPU
   - 更新了所有模型相关文件以使用 `.to(device)` 方法
   - 确保代码可以在不同硬件平台上运行

### 2025/12/22 Peng - 代码简化和环境验证

#### 主要工作内容

1. **实验环境确认**
   - 确认后续实验只在 Nvidia GPU 的个人电脑（Windows 系统）上进行
   - 团队成员均使用 Windows + Nvidia GPU 配置

2. **代码简化**
   - 移除了所有代码中的 `device` 参数和设备选择逻辑
   - 统一使用 CUDA，简化代码结构
   - 所有模型初始化和张量操作默认使用 `.cuda()`

3. **环境验证工具**
   - 创建了 `test_environment.py`，一次性验证三个重要工具：
     - CUDA 可用性和 GPU 信息
     - CLIP 模块导入和可用模型列表
     - Ultralytics 模块导入和 YOLO 类可用性
   - 删除了旧的单独验证脚本，统一使用新的验证工具

### 2025/12/13 Peng - 初始实现和环境配置

#### 主要工作内容

1. **运行环境配置**
   - 创建 conda 虚拟环境（Python 3.9）
   - 安装 PyTorch（CUDA 12.6 支持）
   - 安装项目依赖包和 CLIP 库
   - 配置 IDE 开发环境

2. **模型代码实现**
   - 实现了所有核心模型组件（CLIP 文本编码器、YOLO 视觉骨干、投影网络、跨模态融合模块）
   - 实现了损失函数（蒸馏损失、跨模态对齐损失、特征对齐损失）
   - 完成了主模型类 KDEOVModel 和训练脚本
   - 创建了使用示例代码

3. **代码修复和优化**
   - 修复了 CLIP 库的 JIT 模型加载问题
   - 解决了 IDE 模块导入识别问题
