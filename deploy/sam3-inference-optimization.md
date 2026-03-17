# Feature/Fix: 基于 PyTorch 的 SAM3 HTTP 高性能推理服务优化方案

## 1. 背景（是什么 && 为什么）

### 现状分析
- 当前仓库已经具备 SAM3 图像/视频推理能力，并且内部已经存在若干可复用的性能基础：
- `sam3/model_builder.py` 已提供 `compile=True` 入口。
- `sam3/sam/transformer.py` 已有 `use_fa3` 分支，可接入 FlashAttention-3。
- `sam3/model/sam3_video_predictor.py` 已有 `Sam3VideoPredictorMultiGPU`，支持多进程多 GPU 协同。
- `sam3/model/sam3_video_inference.py`、`sam3/model/sam3_tracking_predictor.py` 已经在关键路径使用 `torch.inference_mode()` 和 BF16 autocast。
- `sam3/model/io_utils.py` 已有异步帧加载、后台线程、TorchCodec 解码逻辑。
- 当前 `deploy/develop.Dockerfile` 与 `deploy/deploy.Dockerfile ` 仍是通用草稿，没有形成面向 HTTP 在线推理的开发容器、生产容器、启动脚本、健康检查、压测和灰度配置。
- 当前仓库没有成型的 HTTP 服务实现，也没有把现有优化能力与更激进的方案如量化、TensorRT、自定义算子、cuDNN/cuBLAS/CUTLASS 路线统一到一份可执行的上线设计中。

### 设计问题
- 现有能力偏“模型内部优化”，缺少“服务级优化编排”，包括批处理、并发、线程、亲和性、预热、缓存、指标和降级。
- 当前部署边界不清晰。用户要求所有新增文件只放在 `deploy`，这意味着服务代码、配置、启动脚本、压测脚本、compose 文件都要集中落在 `deploy`，不能再新增 `service/` 之类目录。
- 当前目标提供 HTTP、WebSocket服务，因此方案应收敛到 FastAPI/Uvicorn/Gunicorn 或等价 HTTP 栈，不再考虑 gRPC、WebSocket 等接口形态。
- 仅依赖 `torch.compile + BF16 + FA3` 虽然已经有明显收益，但在高吞吐场景仍有进一步优化空间，包括：
- 模型量化：权重 INT8、激活 FP8/INT8、分层量化。
- TensorRT：静态 shape 或 shape bucket 下的 detector 子图、tracker 子图引擎化。
- 自定义算子：针对 prompt 编码、mask 后处理、RoPE attention、memory gather/scatter、NMS/阈值化等热点。
- 低层库优化：显式利用 cuDNN、cuBLASLt、CUTLASS、Triton kernel。

## 2. 目标（想做成什么样子）

### 核心目标
- 在不改变 SAM3 主要功能语义的前提下，构建一套面向 HTTP 在线推理的高性能部署方案。
- 所有新增文件只放在 `deploy` 目录；如需改造模型或算子，只修改 `sam3/**` 现有文件，不新增 `deploy` 之外的新文件。
- 同时规划“短期可落地优化”和“中长期高收益优化”，避免方案只停留在 PyTorch 默认路径。

### 预期效果
- 功能层面：提供标准 HTTP 接口，覆盖图像推理、视频会话推理、流式传播、关闭会话、健康检查、指标拉取。
- 性能层面：先通过 `torch.compile + BF16 + FA3 + 多进程/多 GPU + 异步解码 + 微批` 获得主收益；后续可通过量化、TensorRT、自定义算子进一步突破。
- 工程层面：开发容器和生产容器分离；同一套配置可以在单卡、多卡、不同 GPU 型号间切换；所有激进优化都具备独立开关和回退路径。

## 3. 范围明确（需要修改与新增那些文件,不能修改和新增那些文件）

### 需要修改的文件
- `deploy/develop.Dockerfile` - 构建开发容器，加入 CUDA/PyTorch/编译工具链/分析工具/HTTP 服务依赖。
- `deploy/deploy.Dockerfile ` - 构建生产容器，固定依赖，加入 HTTP 服务启动、健康检查和预热。
- `deploy/README.md` - 补充开发、压测、上线、回滚说明。
- `sam3/model_builder.py` - 统一暴露 `use_fa3`、`compile_mode`、`amp_dtype`、`engine_backend`、`quant_mode`、`warmup_shapes` 等参数。
- `sam3/sam/transformer.py` - 增强 FA3、SDPA、TensorRT plugin、自定义 attention kernel 的切换能力。
- `sam3/model/sam3_video_predictor.py` - 增强 HTTP 服务场景所需的 worker 生命周期、健康探针、CPU 亲和和超时控制。
- `sam3/model/io_utils.py` - 增强解码线程、预取、GPU decode、pinned memory、非阻塞 H2D。
- `sam3/model/sam3_video_inference.py` - 增加 engine 化、warmup shape bucket、量化和 TensorRT fallback 入口。
- `sam3/model/decoder.py`、`sam3/model/encoder.py`、`sam3/model/vitdet.py` - 视热点情况增加 `torch.compile`、Triton、自定义 kernel 或 TensorRT 友好的改造。

### 需要新增的文件
- `deploy/sam3-inference-optimization.md` - 本优化方案文档。
- `deploy/http_service.py` - HTTP 服务主入口，负责路由、模型初始化、请求分发。
- `deploy/http_schemas.py` - HTTP 请求/响应 schema。
- `deploy/http_runtime.py` - 模型运行时、队列、微批、线程池、显存治理。
- `deploy/http_image_service.py` - 图像推理封装。
- `deploy/http_video_service.py` - 视频会话推理封装。
- `deploy/http_metrics.py` - Prometheus 指标与结构化日志埋点。
- `deploy/http_warmup.py` - 启动期预热、compile cache、TRT engine cache 构建。
- `deploy/http_config.py` - 环境变量配置中心。
- `deploy/http_threading.py` - 线程池、affinity、NUMA、任务队列封装。
- `deploy/docker-compose.dev.yml` - 开发容器编排。
- `deploy/docker-compose.prod.yml` - 生产容器编排。
- `deploy/.env.example` - 环境变量模板。
- `deploy/benchmark_http_image.py` - 图像 HTTP 压测脚本。
- `deploy/benchmark_http_video.py` - 视频 HTTP 压测脚本。
- `deploy/smoke_test_http.py` - 烟测脚本。
- `deploy/build_trt_engine.py` - TensorRT engine 构建脚本。
- `deploy/export_onnx.py` - ONNX/TensorRT 导出脚本。
- `deploy/calibrate_int8.py` - 量化校准脚本。

### 不能修改的文件
- `sam3/train/**` - 原因：训练链路不纳入此次 HTTP 推理服务改造。
- `examples/**` - 原因：示例 notebook 不承担生产逻辑。
- `README.md` - 原因：保持上游说明简洁。
- 非 `deploy` 目录下的新文件 - 原因：本次硬约束是所有新增文件只写在 `deploy`。

### 影响范围
- 直接影响：模型构建、HTTP 服务、容器镜像、性能分析、压测、上线流程。
- 间接影响：业务网关、对象存储、Prometheus/Grafana、K8s/容器编排平台、CI/CD 镜像构建流程。

## 4. 设计方案（大致的方案）

### 新旧方案对比
- **旧方案**：本地 Python/Notebook 推理 + 零散部署草稿，缺少服务层统一编排。
- **新方案**：以 HTTP 服务为唯一对外协议，在 `deploy` 内集中放置服务代码、配置、压测脚本、容器编排和引擎导出工具；在 `sam3/**` 中做必要性能改造。
- **核心变化**：从“模型可跑”升级为“HTTP 服务可开发、可压测、可观测、可灰度、可多路线优化”。

### 架构/接口设计
- 服务协议固定为 HTTP，建议采用 `FastAPI + Uvicorn`。
- 路由设计：
- `POST /v1/image/infer`
- `POST /v1/video/session/start`
- `POST /v1/video/session/prompt`
- `POST /v1/video/session/propagate`
- `POST /v1/video/session/close`
- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- 服务进程模型建议分层：
- 图像服务：单机多进程，每进程独占 1 张 GPU，进程内做动态微批。
- 视频服务：单实例独占多卡，复用 `Sam3VideoPredictorMultiGPU` 处理会话型状态。
- 统一配置项：
- 模型后端：`pytorch_eager | pytorch_compile | tensorrt`
- 注意力后端：`sdpa | fa3 | custom_kernel`
- 精度模式：`fp32 | tf32 | bf16 | fp16 | int8_weight_only | int8 | fp8`
- 解码后端：`cv2 | torchcodec_cpu | torchcodec_gpu`
- 并行模式：`single_gpu | multi_proc | multi_gpu_video`

### 核心操作流程
1. 容器启动，加载环境变量和 GPU 拓扑信息。
2. HTTP 服务初始化时根据配置选择图像/视频服务实例和后端。
3. 若后端为 `pytorch_compile`，则执行 compile 预热。
4. 若后端为 `tensorrt`，则加载现成 engine；若 engine 不存在，则启动前由 `deploy/build_trt_engine.py` 离线构建。
5. 图像请求进入分辨率桶和 prompt 类型桶，进行动态微批。
6. 视频请求按 session 维护状态，单 session 不跨 worker 迁移，避免状态同步开销。
7. 对输入视频优先异步解码、预取和 pinned memory 传输。
8. 输出结果时记录耗时拆解、显存、队列长度、会话数、TRT 命中率、compile 回退率。

### 技术细节

#### 4.1 第一层优化：纯 PyTorch 高收益低风险路径
- `torch.compile`
  - 图像 detector 路径优先编译 vision backbone、transformer encoder/decoder、segmentation head。
  - 视频 tracker 路径继续沿用组件级编译，而不是整模型全图编译。
  - 对动态 shape 强的路径引入 shape bucket，例如 `672/896/1008/1344`，减少重新编译。
  - 启用 compile cache，并在容器启动时预热典型 prompt 数量和典型分辨率。
- FlashAttention-3 / SDPA
  - H100/H200 优先 FA3。
  - A100/L40S/4090 等若 FA3 兼容性不足，则回退到 PyTorch SDPA。
  - 对 `sam3/sam/transformer.py` 增加后端探测、日志和强制回退开关。
- BF16/FP16/TF32
  - Hopper/Ada 优先 BF16。
  - 卷积/GEMM 路径显式启用 TF32：
  - `torch.backends.cuda.matmul.allow_tf32 = True`
  - `torch.backends.cudnn.allow_tf32 = True`
  - 对后处理保持 FP32 或 bool，避免数值漂移。
- 并行与批处理
  - 图像请求：多进程 + 动态微批。
  - 视频请求：会话独占 + 多 GPU predictor。
  - 不推荐在线推理使用 `torch.nn.DataParallel`。
- 多线程与异步 I/O
  - 解码线程池、下载线程池、结果编码线程池分离。
  - 限制 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`torch.set_num_threads`，避免线程争抢。
  - 对图片下载、视频下载、帧加载使用 bounded queue，防止 OOM。

#### 4.2 第二层优化：PyTorch 内部重写热点算子
- Triton kernel
  - 适合改写轻量但高频的张量重排、mask threshold、small-reduction、gather/scatter。
  - 优先改造：
  - prompt/mask 后处理中的逐元素与 reduction 逻辑。
  - tracker memory 读写中的 scatter/gather。
  - shape 固定的小型融合 kernel。
- 自定义 CUDA/C++ 算子
  - 对高频热点可新增 `torch.ops.sam3_*` 风格算子，但由于不能新增 `deploy` 外文件，本轮只在方案中列为可选，需要后续评估是否允许在 `sam3` 下新增扩展文件。
  - 优先候选：
  - 自定义 RoPE + QKV 变换融合。
  - 二值 mask 后处理与 box 提取融合。
  - 多对象 tracker 结果聚合和过滤融合。
- cuDNN / cuBLASLt / CUTLASS
  - cuDNN：用于卷积、归一化、部分 attention 子路径的高性能实现选择。
  - cuBLASLt：用于 GEMM autotune、epilogue fusion、BF16/FP8 matmul 优化。
  - CUTLASS：用于自定义 GEMM/attention/MLP kernel，适合 TensorRT plugin 或自定义 op 方案。
  - 实施方式：
  - 先用 profiler 确认 GEMM/attention 是否是主热点。
  - 如果热点集中在线性层/attention，则优先考虑 cuBLASLt autotune 或 CUTLASS plugin。

#### 4.3 第三层优化：模型量化
- 权重量化
  - 起步方案：weight-only INT8，适合线性层为主的 detector/transformer。
  - 优点：开发风险低、内存收益快、可与 BF16 激活共存。
- 激活量化
  - PTQ INT8：需要离线校准集，适用于 detector 子图优先。
  - QAT 不建议纳入本期，因为改动训练链路。
- FP8
  - 对 Hopper 平台可评估 FP8 路径，但工程复杂度高于 BF16/INT8。
- 量化策略建议
  - 第一优先：只量化 detector 的线性层和部分 MLP。
  - 第二优先：量化 tracker 的 memory encoder/decoder。
  - 不建议首期量化最终 mask 后处理和几何逻辑。
- 量化风险
  - prompt 稀疏场景和长视频跟踪对数值敏感。
  - 必须建立精度门槛：mask IoU、box AP、视频跟踪稳定性。

#### 4.4 第四层优化：TensorRT 推理
- 适用范围
  - 图像 detector 更适合先导出 ONNX/TensorRT。
  - 视频 tracker 因状态复杂、动态 shape 多，建议按子图逐步 engine 化，而不是一次性整模型 TRT。
- 推荐路线
  - 路线 A：只将视觉 backbone + detector transformer + segmentation head 导出 TRT。
  - 路线 B：将 tracker 中固定 shape 的 decoder 子图导出 TRT，其余仍走 PyTorch。
  - 路线 C：构建混合后端，PyTorch 负责状态管理，TRT 负责纯计算密集型子图。
- shape 策略
  - 使用固定 shape 或 bucket shape 构建多个 engine。
  - 图像建议按 `672/896/1008/1344` 分桶。
  - 视频建议按对象数 bucket、帧 size bucket 做有限集合，而非完全动态 shape。
- TensorRT plugin
  - 如 attention、RoPE、特殊 gather/scatter 不易原生导出，可通过 plugin 实现。
  - plugin 底层可基于 CUTLASS 或自定义 CUDA kernel。
- TRT 风险
  - 导出图稳定性。
  - engine 构建时间长。
  - 版本耦合强，需固定 CUDA/TensorRT/驱动版本。

#### 4.5 第五层优化：容器、系统与运行时
- 开发容器
  - 保留 profiler、`nsys`、`ncu`、`py-spy`、`perf`、测试依赖、Jupyter 能力。
  - 允许在容器内构建 TRT engine、校准量化参数、跑 microbenchmark。
- 生产容器
  - 精简运行时，只保留 HTTP 服务、模型权重缓存、engine 缓存、必要动态库。
  - 非 root 运行。
  - 启动前完成健康自检：GPU、驱动、权重、engine、FA3、compile cache。
- CUDA 运行时参数
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - `CUDA_DEVICE_MAX_CONNECTIONS=1` 或针对 workload 调优
  - `NCCL_P2P_DISABLE=0`
  - `NCCL_IB_DISABLE` 按机器网络情况配置
- CPU/NUMA
  - 每个 HTTP worker 绑核。
  - GPU 与 CPU NUMA 节点对齐。
  - 解码线程与模型线程分组。

#### 4.6 第六层优化：缓存与复用
- compile cache：启动前构建，跨容器复用。
- TRT engine cache：持久化到挂载卷。
- tokenizer/text encoding cache：相同 prompt 直接命中。
- 图像预处理 cache：同图重复推理时复用归一化结果。
- 视频 session cache：冷热分层，限制最大会话数和空闲 TTL。

## 5. 实施步骤

### Step 1: 收敛 `deploy` 目录边界与 HTTP 服务骨架
- 明确所有新增文件只能放在 `deploy`。
- 新增 `deploy/http_service.py`、`deploy/http_runtime.py`、`deploy/http_config.py` 等 HTTP 服务骨架。
- 禁止新增 `deploy` 之外的新服务目录。
- 涉及文件清单：
- `deploy/http_service.py`
- `deploy/http_runtime.py`
- `deploy/http_config.py`
- `deploy/http_schemas.py`
- `deploy/http_image_service.py`
- `deploy/http_video_service.py`

### Step 2: 改造容器与启动流程
- 修正 `deploy/deploy.Dockerfile ` 文件名尾随空格问题，规范化为生产 Dockerfile。
- 开发容器支持 profiler、量化校准、TRT 导出和压测。
- 生产容器仅保留运行时依赖与 HTTP 服务入口。
- 启动顺序：环境检查 -> 权重检查 -> engine 检查 -> compile/engine 预热 -> 标记 `readyz`。
- 涉及文件清单：
- `deploy/develop.Dockerfile`
- `deploy/deploy.Dockerfile `
- `deploy/docker-compose.dev.yml`
- `deploy/docker-compose.prod.yml`
- `deploy/.env.example`

### Step 3: 打通第一层 PyTorch 优化
- 在 `sam3/model_builder.py` 统一新增后端和精度参数。
- 在 `sam3/sam/transformer.py` 打通 `sdpa/fa3/custom_kernel` 切换。
- 在 `sam3/model/io_utils.py` 增加线程、预取、GPU decode 配置。
- 图像路径接入动态微批；视频路径接入会话数限制和多 GPU 配置。
- 涉及文件清单：
- `sam3/model_builder.py`
- `sam3/sam/transformer.py`
- `sam3/model/io_utils.py`
- `sam3/model/sam3_video_predictor.py`
- `sam3/model/sam3_video_inference.py`
- `deploy/http_runtime.py`

### Step 4: 做性能剖析并选择热点改造
- 使用 `torch.profiler`、`nsys`、`ncu` 确认热点。
- 若热点主要在 attention/GEMM，则优先 FA3、cuBLASLt、CUTLASS、TensorRT。
- 若热点主要在 scatter/gather/postprocess，则优先 Triton 或自定义 kernel。
- 输出热点表和优化优先级。
- 涉及文件清单：
- `deploy/benchmark_http_image.py`
- `deploy/benchmark_http_video.py`
- `deploy/README.md`

### Step 5: 引入量化
- 先做 detector 的 weight-only INT8。
- 再评估 PTQ INT8。
- 建立校准集、导出流程和精度验收标准。
- 量化路径必须支持按环境变量一键关闭。
- 涉及文件清单：
- `deploy/calibrate_int8.py`
- `deploy/http_runtime.py`
- `sam3/model_builder.py`

### Step 6: 引入 TensorRT
- 先导出 detector 子图 ONNX。
- 离线构建 TRT engine 并缓存。
- 服务运行时支持 `PyTorch/TRT/Hybrid` 三种后端。
- 对无法导出的子图保留 PyTorch fallback。
- 涉及文件清单：
- `deploy/export_onnx.py`
- `deploy/build_trt_engine.py`
- `deploy/http_runtime.py`
- `sam3/model_builder.py`
- `sam3/sam/transformer.py`

### Step 7: 引入更底层算子优化
- 在明确热点后评估 Triton、自定义 CUDA op、CUTLASS plugin。
- 先做 ROI 最大的 1 到 2 个算子，不做大面积重写。
- 若需新增 C++/CUDA 源码，需单独确认是否放宽“新增文件只在 `deploy`”限制。
- 涉及文件清单：
- `sam3/**` 相关热点文件
- `deploy/README.md`

### 兼容性与迁移
- 向后兼容策略：保持原有 Python 调用方式不变，新增参数都有默认值。
- 迁移路径：先上纯 PyTorch HTTP 服务，再灰度量化，再灰度 TRT，再灰度自定义 kernel。
- 过渡期安排：任何激进优化都必须保留环境变量开关和 PyTorch fallback。

## 6. 验收标准

### 功能测试
- 测试用例 1：图像 HTTP 推理返回结构正确。
- 测试用例 2：视频 HTTP 会话全流程正确。
- 测试用例 3：关闭 compile 后服务可正常运行。
- 测试用例 4：关闭 FA3 后回退到 SDPA 正常。
- 测试用例 5：开启 TRT 后命中 engine 正常；engine 缺失时回退正常。
- 测试用例 6：开启 INT8 量化后服务可运行，输出不出现明显异常。

### 代码质量
- 配置项、环境变量、HTTP 协议、压测命令均有文档。
- 所有优化项具备显式开关。
- 指标、日志、健康检查齐全。

### 回归测试
- 现有 SAM3 图像和视频推理路径在默认配置下行为不变。
- 在不启用任何高级优化时，HTTP 服务仍可稳定运行。
- 修改 `sam3/**` 后不破坏原生本地推理接口。

### 性能与可维护性
- 图像服务：
- 稳态吞吐较原始 eager 模式显著提升。
- P95 延迟明显下降，首请求抖动被预热吸收。
- 视频服务：
- 多 GPU 吞吐相较单 GPU 有明确提升。
- 长视频 session 的显存/CPU 内存增长可控。
- 量化/TRT：
- 在满足精度门槛前提下，相比纯 PyTorch 获得额外吞吐收益或显存收益。
- 可维护性：
- 所有高级路线都可独立启停。
- 容器、服务、引擎、校准、压测都集中在 `deploy` 管理。

### 文档与示例
- `deploy/README.md` 必须包含：
- 开发容器启动方法。
- 生产容器启动方法。
- HTTP API 示例。
- 压测命令。
- 量化和 TRT 构建流程。
- 回滚方法。

## 7. 其他说明

### 相关资源
- 当前可复用实现：
- `sam3/model_builder.py`
- `sam3/sam/transformer.py`
- `sam3/model/sam3_video_predictor.py`
- `sam3/model/sam3_video_inference.py`
- `sam3/model/io_utils.py`
- 当前部署草稿：
- `deploy/develop.Dockerfile`
- `deploy/deploy.Dockerfile `

### 风险与应对
- 潜在风险 1：`torch.compile` 对动态 shape 敏感，可能出现频繁重编译。
- 应对措施：shape bucket + 启动预热 + compile cache + 一键关闭。
- 潜在风险 2：FA3、TensorRT、FP8、INT8 都与 GPU 架构和软件版本强耦合。
- 应对措施：按 GPU 白名单启用，环境检查失败自动回退。
- 潜在风险 3：量化可能伤害长视频跟踪稳定性。
- 应对措施：先 detector 后 tracker；先 weight-only 后 activation quant；建立精度基线。
- 潜在风险 4：TRT 导出覆盖率不足，导致混合后端复杂度高。
- 应对措施：先做 detector 子图，不追求首期全量 TRT。
- 潜在风险 5：若后续必须新增 CUDA/C++ 源码，自定义算子无法满足“新增文件仅在 `deploy`”限制。
- 应对措施：当前先以 Triton、Torch 自定义 op 包装、TensorRT plugin 方案为主；若收益足够高，再单独申请放宽限制。
- 潜在风险 6：当前生产 Dockerfile 文件名尾随空格，容易导致 CI/CD 或构建命令失误。
- 应对措施：优先修正文件名并统一引用。

### 依赖关系
- 依赖的其他 Feature：
- Hugging Face 权重访问权限。
- 稳定的 NVIDIA 驱动、CUDA、TensorRT 运行时。
- 可用的 Prometheus/Grafana。
- 被依赖的场景：
- 图像在线分割服务。
- 视频在线跟踪服务。
- 低时延交互式标注平台。
