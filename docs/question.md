
## torch.compile

- 有几种模式
```
# Transformer 编译
transformer = TransformerEncoder(
    compile_mode="max-autotune",  # 指定编译模式
    ...
)

# 文本编码器编译
text_encoder = VETextEncoder(
    compile_mode="default",  # 使用默认模式
    ...
)
```
- 可以针对全图使用吗

## 已有高效的算子，采用高效的算子可以吗？

- 如何开启fa3呢

## 整个模型跑下来需要多少内存，单卡可以跑下来吗，目前甲方是什么机器

## 假如单卡可以跑下来，data parallel是最快，那如何做到数据同步呢



model = build_sam3_video_model(
    compile=True,
    compile_mode="max-autotune",
    use_fa3=True,  # 如果可用
)


## 如何采用fp16计算