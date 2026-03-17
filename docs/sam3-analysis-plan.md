# SAM 3 技术解析计划

## 项目概述

SAM 3 (Segment Anything with Concepts) 是 Meta 发布的统一基础模型，用于图像和视频中的可提示分割。它能够使用文本或视觉提示（如点、框和掩码）来检测、分割和跟踪对象。

**关键特性：**
- 848M 参数规模的检测器-跟踪器架构
- 支持 270K+ 开放词汇概念
- 检测器与跟踪器解耦设计
- Presence Token 机制提升语义区分能力
- 支持多 GPU 分布式推理

## 分析大纲

### 1. 核心架构总览
- [ ] SAM 3 整体架构设计
- [ ] 检测器-跟踪器解耦机制
- [ ] 数据流向与调用链路

### 2. 视觉编码器 (Visual Encoder)
- [ ] ViT Backbone 结构 (`sam3/model/vitdet.py`)
- [ ] Sam3DualViTDetNeck 特征金字塔 (`sam3/model/necks.py`)
- [ ] 位置编码机制 (`sam3/model/position_encoding.py`)
- [ ] 混合注意力机制（全局+窗口）

### 3. 文本编码器 (Text Encoder)
- [ ] VETextEncoder 架构 (`sam3/model/text_encoder_ve.py`)
- [ ] SimpleTokenizer 分词器 (`sam3/model/tokenizer_ve.py`)
- [ ] 文本-视觉融合机制 (`sam3/model/vl_combiner.py`)

### 4. 检测器模块 (Detector)
- [ ] Transformer 编码器 (`sam3/model/encoder.py`)
- [ ] Transformer 解码器 (`sam3/model/decoder.py`)
- [ ] 几何提示编码器 (`sam3/model/geometry_encoders.py`)
- [ ] 分割头 (`sam3/model/maskformer_segmentation.py`)
- [ ] Presence Token 机制

### 5. 跟踪器模块 (Tracker)
- [ ] Sam3TrackerPredictor 核心逻辑 (`sam3/model/sam3_tracking_predictor.py`)
- [ ] 内存编码机制 (`sam3/model/memory.py`)
- [ ] RoPE 注意力机制 (`sam3/sam/transformer.py`)
- [ ] 时空建模策略

### 6. 图像分割流程
- [ ] Sam3Image 模型 (`sam3/model/sam3_image.py`)
- [ ] Sam3Processor 处理器 (`sam3/model/sam3_image_processor.py`)
- [ ] 点/框提示编码 (`sam3/sam/prompt_encoder.py`)
- [ ] 掩码解码器 (`sam3/sam/mask_decoder.py`)

### 7. 视频跟踪流程
- [ ] Sam3VideoPredictor 视频推理 (`sam3/model/sam3_video_predictor.py`)
- [ ] Sam3VideoInference 视频推理引擎 (`sam3/model/sam3_video_inference.py`)
- [ ] 多 GPU 通信机制
- [ ] 时空关联策略

### 8. Agent 系统
- [ ] Agent 核心逻辑 (`sam3/agent/agent_core.py`)
- [ ] LLM 集成 (`sam3/agent/client_llm.py`)
- [ ] 推理流程 (`sam3/agent/inference.py`)
- [ ] 辅助工具（zoom_in, masks, boxes 等）

### 9. 性能优化模块
- [ ] FA3 优化注意力 (`sam3/perflib/fa3.py`)
- [ ] 编译优化 (`sam3/perflib/compile.py`)
- [ ] Triton 算子 (`sam3/perflib/triton/`)
- [ ] 掩码操作优化 (`sam3/perflib/masks_ops.py`)

### 10. 训练与评估
- [ ] 数据预处理 (`sam3/train/data/`)
- [ ] 损失函数 (`sam3/train/loss/`)
- [ ] 评估指标（cgF1, HOTA 等）

## 进度追踪表

| 模块 | 状态 | 文档路径 | 完成时间 |
|------|------|----------|----------|
| 分析计划 | ✅ | sam3-analysis-plan.md | 2026-03-16 |
| 核心架构总览 | ✅ | modules/sam3-module-01-overview.md | 2026-03-16 |
| 视觉编码器 | ✅ | modules/sam3-module-02-visual-encoder.md | 2026-03-16 |
| 文本编码器 | ✅ | modules/sam3-module-03-text-encoder.md | 2026-03-16 |
| 检测器模块 | ✅ | modules/sam3-module-04-detector.md | 2026-03-16 |
| 跟踪器模块 | ✅ | modules/sam3-module-05-tracker.md | 2026-03-16 |
| 图像分割流程 | ✅ | modules/sam3-module-06-image-seg.md | 2026-03-16 |
| 视频跟踪流程 | ✅ | modules/sam3-module-07-video-tracking.md | 2026-03-16 |
| Agent 系统 | ✅ | modules/sam3-module-08-agent.md | 2026-03-16 |
| 性能优化模块 | ✅ | modules/sam3-module-09-perf-optim.md | 2026-03-16 |
| 训练与评估 | ✅ | modules/sam3-module-10-training.md | 2026-03-16 |
| 整合文档 | ✅ | sam3-technical-analysis.md | 2026-03-16 |

## 技术要点

### 关键设计决策

1. **检测器-跟踪器解耦**: 最小化任务干扰，提高可扩展性
2. **Presence Token**: 改善相关文本提示之间的区分度
3. **数据引擎**: 自动标注 400万+ 独特概念
4. **多 GPU 分布式推理**: 支持大规模视频处理

### 性能瓶颈与优化

1. **注意力计算优化**: FA3 + Triton
2. **激活检查点**: 减少显存占用
3. **算子融合**: 提升计算效率
4. **通信优化**: 减少多 GPU 同步开销

## 文档结构

```
docs/
├── sam3-analysis-plan.md          # 本文件 - 分析计划
├── sam3-technical-analysis.md      # 最终整合文档
├── modules/
│   ├── sam3-module-01-overview.md      # 核心架构总览
│   ├── sam3-module-02-visual-encoder.md # 视觉编码器
│   ├── sam3-module-03-text-encoder.md  # 文本编码器
│   ├── sam3-module-04-detector.md      # 检测器模块
│   ├── sam3-module-05-tracker.md       # 跟踪器模块
│   ├── sam3-module-06-image-seg.md     # 图像分割流程
│   ├── sam3-module-07-video-tracking.md # 视频跟踪流程
│   ├── sam3-module-08-agent.md         # Agent 系统
│   ├── sam3-module-09-perf-optim.md    # 性能优化
│   └── sam3-module-10-training.md      # 训练与评估
└── diagrams/
    ├── architecture.png           # 整体架构图
    ├── data-flow.png              # 数据流向图
    ├── detector.png               # 检测器架构
    ├── tracker.png                # 跟踪器架构
    └── multi-gpu.png              # 多 GPU 通信
```

## Checkpoint 机制

每个模块分析完成后：
1. 更新本计划文件的进度表
2. 保存对应的模块文档
3. 更新 MEMORY.md 以记录关键发现

如分析中断，可根据本计划恢复进度。
