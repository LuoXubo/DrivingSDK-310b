# FlashOCC car 数据集训练说明

本文档记录如何在 **car_perception_grid** 自定义数据集上，使用 DrivingSDK 8.3 + 昇腾 NPU 完成 FlashOCC 训练。目标是：以后重新训练时，按本文步骤即可复现全流程。

---

## 1. 一句话流程

```text
原始数据 → create_car_perception_flashocc.py 转换
        → link_car_perception_media.py 硬链媒体
        → fix / visualize 校验
        → NPU 环境 + bevdet 预训练权重
        → 冒烟训练
        → 正式多卡训练
        → eval_car_grid_occ.py 评测
```

---

## 2. 关键路径

| 用途 | 路径 |
|------|------|
| FlashOCC 工程根目录 | `DrivingSDK/model_examples/FlashOCC/FlashOCC/` |
| 容器内工程根目录 | `/workspace/flashocc_cann83/DrivingSDK/model_examples/FlashOCC/FlashOCC` |
| 原始数据（宿主机） | `/data/car_perception_grid` |
| 转换后训练数据 | `data/car_perception_grid/nuscenes/` |
| 训练配置 | `projects/configs/flashocc/flashocc-r50-car-grid.py` |
| 预训练权重 | `ckpts/bevdet-r50-cbgs.pth` |
| 推荐训练产物 | `work_dirs/car_grid_v4/epoch_12_ema.pth` |

---

## 3. 环境与容器

### 3.1 进入 Docker

使用已创建的 DrivingSDK 8.3 容器（详见 `docs/runwithdocker.md`）：

```bash
docker start flashocc_drivingsdk_83
docker exec -it flashocc_drivingsdk_83 /bin/bash
```

若需容器内直接读原始数据，创建容器时增加挂载：

```bash
-v /data/car_perception_grid:/data/car_perception_grid
```

当前默认方案是在宿主机执行 `link_car_perception_media.py`，把图片/深度硬链到工程 `samples/` 下，**不依赖容器内访问 `/data`**。

### 3.2 激活 conda 与 NPU 环境变量

每次训练前在容器内执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate torch2.1.0_py38

export TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export TORCH_NPU_LIB=$(python -c "import torch_npu, os; print(os.path.join(os.path.dirname(torch_npu.__file__), 'lib'))")
export PY_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export ASCEND_CUSTOM_OPP_PATH=${PY_SITE}/mx_driving/packages/vendors/customize
export LD_LIBRARY_PATH=${ASCEND_CUSTOM_OPP_PATH}/op_api/lib:${TORCH_LIB}:${TORCH_NPU_LIB}:${LD_LIBRARY_PATH}

cd /workspace/flashocc_cann83/DrivingSDK/model_examples/FlashOCC/FlashOCC
export PYTHONPATH=$(pwd)/projects:$(pwd)/mmdetection3d:$(pwd):$PYTHONPATH
```

### 3.3 选择 NPU

单卡训练：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
```

多卡训练（例如 4 卡）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
```

---

## 4. 数据准备

### 4.1 原始数据格式

原始目录结构：

```text
/data/car_perception_grid/
├── near_20260610_224039/poses/<pose_name>/
└── wave1_20260610_092058/poses/<pose_name>/
```

每个 pose 目录下，同一 `stem` 时刻包含：

| 文件 | 说明 |
|------|------|
| `{stem}_cam1.png` / `{stem}_cam2.png` | 左/右前视，1024×1024 灰度 |
| `{stem}_cam1_depth.npy` / `{stem}_cam2_depth.npy` | 深度 float32，无效值约 65504 |
| `{stem}_seg_cam1_raw.png` / `{stem}_seg_cam2_raw.png` | 分割图，B 通道编码 0/1/2 |
| `{stem}_lidar_sensor.ply` | 点云 |
| `{stem}_occ.npz`（少数帧） | 预计算 16 层 OCC GT |

### 4.2 语义标签映射

| 原始 seg 值 | 含义 | OCC 训练标签 |
|------------|------|-------------|
| 0 | 可通行 (passable) | **0** |
| 1 | 车辆障碍 (car) | **1** |
| 2 | 未知 (unknown) | **255**（ignore，不参与 loss） |

### 4.3 转换脚本

在 FlashOCC 根目录执行：

```bash
cd /workspace/flashocc_cann83/DrivingSDK/model_examples/FlashOCC/FlashOCC

python tools/create_car_perception_flashocc.py --src /data/car_perception_grid
```

常用参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--src` | `/data/car_perception_grid` | 原始数据根目录 |
| `--out` | `data/car_perception_grid/nuscenes` | 输出目录 |
| `--val-ratio` | `0.1` | 验证集比例 |
| `--seed` | `42` | 划分随机种子 |
| `--max-samples` | `0` | 限制样本数（0=全部） |
| `--raster-step` | `1` | depth+seg 投影步长，越小 mask 越密 |
| `--rebuild-labels-only` | — | 仅重建 `labels.npz`，不重新扫描原始数据 |
| `--force` | — | 强制重建所有 labels |
| `--compare-mask` | — | 对比 v2/v3 mask 覆盖率 |

转换后目录：

```text
data/car_perception_grid/nuscenes/
├── bevdetv2-nuscenes_infos_train.pkl   # 约 1459 样本
├── bevdetv2-nuscenes_infos_val.pkl     # 约 158 样本
├── samples/
│   ├── CAM_FRONT_LEFT/*.png
│   ├── CAM_FRONT_RIGHT/*.png
│   ├── depth/CAM_FRONT_LEFT/*_depth.npy
│   ├── depth/CAM_FRONT_RIGHT/*_depth.npy
│   └── LIDAR_TOP/*.bin
└── gts/<scene>/<token>/labels.npz
```

每个 `labels.npz` 包含：

- `semantics`：`(200, 200, 2)` uint8，取值 0 / 1 / 255
- `mask_camera`：`(200, 200, 2)` uint8，相机可见区域
- `mask_lidar`：`(200, 200, 2)` uint8，点云命中区域

### 4.4 硬链媒体（Docker 训练必做）

若 pkl 里路径指向宿主机绝对路径，在**宿主机或容器**执行：

```bash
python tools/link_car_perception_media.py
```

作用：把 `/data/car_perception_grid/...` 硬链到 `data/car_perception_grid/nuscenes/samples/`，并重写 pkl 为工程内相对路径。

### 4.5 清洗 pkl

剔除无效样本（seg 误匹配、文件缺失等）：

```bash
python tools/fix_car_perception_pkl.py
```

### 4.6 训练前可视化检查

```bash
python tools/visualize_car_grid_samples.py
```

输出目录：`data/car_perception_grid/vis_train_samples/`

每张 panel 含：双相机图、深度、分割、OCC 两层 BEV、lidar mask。

指定样本：

```bash
python tools/visualize_car_grid_samples.py --indices 0,729,1458 --max-samples 3
```

### 4.7 已知数据问题（训练前必读）

历史统计（转换时）：

| 划分 | 样本数 | 含 car(1) 的样本 | 语义唯一值 |
|------|--------|-----------------|-----------|
| train | 1459 | **0** | [0, 255] |
| val | 158 | **4** | [0, 1, 255] |

**结论**：若按默认顺序划分，train 集完全没有 car 标签，car 类只在 val 的 4 帧出现。要做 3 类 OCC 训练，需要：

1. 重新划分 train/val，保证 train 含 car 样本；或
2. 补充含障碍物的数据；或
3. 调整 `--val-ratio` / `--seed` 后重新转换

camera mask 较稀疏（约 3.6% 体素有标注），与 depth+seg 投影策略有关。

---

## 5. 预训练权重

FlashOCC 从 BEVDet 检测预训练初始化 backbone 和 view transformer：

```bash
ls -lh ckpts/bevdet-r50-cbgs.pth
```

若权重在 `/data/ckpts/`：

```bash
mkdir -p ckpts
ln -sf /data/ckpts/bevdet-r50-cbgs.pth ckpts/bevdet-r50-cbgs.pth
```

加载时会有预期内的 mismatch 警告（depth_net 维度、occ_head 随机初始化等），属于正常现象。

---

## 6. 训练配置说明

配置文件：`projects/configs/flashocc/flashocc-r50-car-grid.py`  
继承自 `flashocc-r50-perf.py`，关键覆盖如下。

### 6.1 数据与相机

| 项 | 值 |
|----|-----|
| 相机 | `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`（2 路） |
| 源图尺寸 | 1024 × 1024 |
| 网络输入 | 256 × 704 |
| BDA 增广 | 全部关闭（rot/scale/flip=恒等） |
| depth 加载 | `PrepareImageInputs(load_depth=True, depth_invalid=60000.0)` |

### 6.2 BEV 栅格

| 项 | 值 |
|----|-----|
| x / y 范围 | [-40, 40] m，步长 0.4 m → 200×200 |
| z 范围 | [-1, 5.4] m |
| depth bins | [0.5, 12.0]，步长 0.5 |
| OCC 高度层 Dz | **2** |
| 语义类别数 | **3** |

### 6.3 模型

| 项 | 值 |
|----|-----|
| 模型类型 | `BEVDetOCC`（纯相机，无 depth loss） |
| backbone | ResNet-50 |
| view transformer | `LSSViewTransformer` |
| occ head | `BEVOCCHead2D` |
| 预训练 | `ckpts/bevdet-r50-cbgs.pth` |

### 6.4 优化器与 schedule

| 项 | 默认值 |
|----|--------|
| optimizer | `NpuFusedAdamW` |
| lr | `2.45e-4` |
| weight_decay | `1e-2` |
| grad_clip | max_norm=5 |
| warmup | linear, 200 iters, ratio=0.001 |
| lr step | epoch 24（car 配置 max_epochs=12 时 step 实际不触发） |
| max_epochs | **12**（配置中覆盖） |
| EMA | `MEGVIIEMAHook` |
| loss 曲线 | `LossCurveHook` → `loss_curves.json` |

### 6.5 数据加载默认值

| 项 | 默认值 |
|----|--------|
| samples_per_gpu | 2 |
| workers_per_gpu | 2 |
| evaluation | 关闭（interval=999） |

---

## 7. 损失函数

当前 car 配置**只有一项训练损失**：

### 7.1 `loss_occ`（CrossEntropyLoss）

```python
loss_occ=dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    ignore_index=255,
    loss_weight=1.0,
    class_weight=[1.0, 15.0, 1.0],  # passable, car, 第3类
)
```

计算规则：

1. 只在 `mask_camera=1` 的体素上计算
2. 标签 255（unknown）被 `ignore_index` 忽略
3. `avg_factor = mask_camera.sum()`，对有效体素平均
4. `class_weight[1]=15.0` 用于缓解 car 类样本极少的问题

**总 loss = loss_occ**（无 depth / lovasz 等辅助项）。

### 7.2 关于 `gt_depth`

pipeline 会加载 `gt_depth`，但 `BEVDetOCC` **不使用 depth 监督**。若要加 depth loss，需改模型为 `BEVDepthOCC` 并使用 `LSSViewTransformerBEVDepth`。

### 7.3 调权重示例

加大 car 类权重：

```bash
--cfg-options model.occ_head.loss_occ.class_weight="[1.0,20.0,1.0]"
```

---

## 8. NPU 代码适配（训练前执行）

`test/train_car_grid_smoke.sh` 内含 idempotent 的 sed 补丁，正式训练前也应执行：

```bash
sed -i 's/^from multiprocessing.dummy import Pool as ThreadPool/# from multiprocessing.dummy import Pool as ThreadPool/' \
  projects/mmdet3d_plugin/models/detectors/bevdet_occ.py || true
sed -i 's/^from ...ops import nearest_assign/# from ...ops import nearest_assign/' \
  projects/mmdet3d_plugin/models/detectors/bevdet_occ.py || true
sed -i 's/^\(\s*\)is_cuda\s*=\s*True/\1is_cuda = False/' \
  projects/mmdet3d_plugin/models/detectors/bevdet_occ.py || true
```

训练入口必须使用 `tools/train.py`（含插件注册 + mx_driving patcher），不要用 mmdetection3d 自带的 train.py。

---

## 9. 冒烟训练（推荐第一步）

验证数据、环境、模型注册是否正常：

```bash
cd /workspace/flashocc_cann83/DrivingSDK/model_examples/FlashOCC/FlashOCC

# 1 卡，batch=1，1 epoch，仅 2 个样本
bash test/train_car_grid_smoke.sh 1 1
```

多卡冒烟：

```bash
bash test/train_car_grid_smoke.sh 4 1
```

脚本自动完成：检查/转换数据 → 硬链媒体 → 生成 2 样本 smoke pkl → NPU 适配 → 训练。

期望日志类似：

```text
Epoch [1][1/2]  loss_occ: 0.8630, loss: 0.8630
Epoch [1][2/2]  loss_occ: 0.8623, loss: 0.8623
Saving checkpoint at 1 epochs
```

输出目录：`work_dirs/car_grid_smoke/`

---

## 10. 正式训练

### 10.1 单卡训练

```bash
export ASCEND_RT_VISIBLE_DEVICES=0

python tools/train.py \
  projects/configs/flashocc/flashocc-r50-car-grid.py \
  --work-dir work_dirs/car_grid_full \
  --gpu-id 0 \
  --cfg-options \
    data.samples_per_gpu=2 \
    data.workers_per_gpu=2 \
    runner.max_epochs=24
```

### 10.2 多卡分布式训练（推荐）

以 **4 卡 × batch 2 = 全局 batch 8** 为例（与 `car_grid_v4` 一致）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

bash tools/dist_train.sh \
  projects/configs/flashocc/flashocc-r50-car-grid.py \
  4 \
  --work-dir work_dirs/car_grid_v4 \
  --cfg-options \
    data.samples_per_gpu=2 \
    data.workers_per_gpu=2 \
    runner.max_epochs=12
```

`dist_train.sh` 内部调用 `tools/train.py --launcher pytorch`。

每 epoch iter 数 ≈ `ceil(train_samples / (num_gpus × samples_per_gpu))`  
例如 1459 样本、4 卡、batch 2 → 182 iter/epoch。

### 10.3 从 checkpoint 恢复

```bash
python tools/train.py \
  projects/configs/flashocc/flashocc-r50-car-grid.py \
  --work-dir work_dirs/car_grid_v4 \
  --resume-from work_dirs/car_grid_v4/epoch_6.pth \
  --gpu-id 0
```

或自动恢复最新 checkpoint：

```bash
python tools/train.py ... --auto-resume
```

### 10.4 常用 `--cfg-options`

| 参数 | 示例 | 说明 |
|------|------|------|
| `data.samples_per_gpu` | `2` | 每卡 batch size |
| `data.workers_per_gpu` | `2` | DataLoader 进程数 |
| `runner.max_epochs` | `12` | 训练 epoch 数 |
| `optimizer.lr` | `1e-4` | 学习率 |
| `model.occ_head.loss_occ.class_weight` | `"[1.0,20.0,1.0]"` | 类别权重 |
| `data.train.ann_file` | 自定义 pkl 路径 | 换训练集 |

---

## 11. 训练监控与产物

### 11.1 日志

训练日志在 `work_dir/` 下：

```text
work_dirs/car_grid_v4/
├── 20260615_121748.log          # 文本日志
├── 20260615_121748.log.json       # JSON 日志
├── flashocc-r50-car-grid.py       # 本次训练完整配置快照
├── loss_curves.json               # LossCurveHook 导出
├── epoch_1.pth / epoch_1_ema.pth
├── ...
└── epoch_12_ema.pth               # 推荐使用 EMA 权重
```

### 11.2 关注指标

训练日志中主要字段：

```text
loss_occ    # OCC 交叉熵损失（也是总 loss）
loss        # 等于 loss_occ
lr          # 当前学习率
grad_norm   # 梯度范数
```

正常训练时 `loss_occ` 应从 ~0.78 逐步下降。

### 11.3 导出 loss 曲线

训练结束后：

```bash
python tools/save_train_loss_curves.py \
  --log-json work_dirs/car_grid_v4/20260615_121748.log.json \
  --work-dir work_dirs/car_grid_v4
```

---

## 12. 训练后评测

使用 val 集评测并可视化 pred vs GT：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0

bash test/eval_car_grid.sh \
  work_dirs/car_grid_v4/epoch_12_ema.pth \
  work_dirs/car_grid_test_results
```

或手动：

```bash
python tools/eval_car_grid_occ.py \
  --checkpoint work_dirs/car_grid_v4/epoch_12_ema.pth \
  --out-dir work_dirs/car_grid_test_results \
  --gpu-id 0
```

输出：

- `metrics.json` — mIoU 等
- `vis_panels/` — 对比图
- `vis_detailed/` — 详细 GT/Pred 面板

实机部署输入输出说明见：`docs/flashocc_car_grid_实机测试IO说明.md`

---

## 13. 常见问题

| 问题 | 处理 |
|------|------|
| `BEVDetOCC is not in the models registry` | 使用 `tools/train.py`，确保 `PYTHONPATH` 含 `projects/` |
| Docker 内找不到图片/深度 | 执行 `python tools/link_car_perception_media.py` |
| 灰度图 normalize 报错 | 已在 `loading.py` 转 RGB，确认代码未回退 |
| `libc10.so` / `libtorch_npu.so` 找不到 | 设置 `LD_LIBRARY_PATH`（见 §3.2） |
| `BEVPoolV3` 不支持 | 检查 `ASCEND_CUSTOM_OPP_PATH`，见 `docs/runwithdocker.md` |
| train 无 car 标签 | 重新划分 train/val 或补充数据（见 §4.7） |
| depth_net shape mismatch 警告 | 正常，car 配置 depth bins 与 nuScenes 预训练不同 |
| OOM | 减小 `data.samples_per_gpu` 或 `with_cp=True` |

---

## 14. 相关脚本与文档索引

| 文件 | 用途 |
|------|------|
| `tools/create_car_perception_flashocc.py` | 原始数据 → FlashOCC 格式 |
| `tools/link_car_perception_media.py` | 硬链媒体 + 修正 pkl 路径 |
| `tools/fix_car_perception_pkl.py` | 清洗无效样本 |
| `tools/visualize_car_grid_samples.py` | 训练数据可视化 |
| `tools/train.py` | NPU 训练入口 |
| `tools/dist_train.sh` | 多卡分布式训练 |
| `test/train_car_grid_smoke.sh` | 冒烟训练一键脚本 |
| `test/eval_car_grid.sh` | 评测一键脚本 |
| `tools/eval_car_grid_occ.py` | 批量推理 + mIoU |
| `projects/configs/flashocc/flashocc-r50-car-grid.py` | car 训练配置 |
| `docs/car_perception_grid_适配说明.md` | 数据适配技术细节 |
| `docs/flashocc_car_grid_实机测试IO说明.md` | 推理/实机部署 |
| `docs/runwithdocker.md` | Docker 环境搭建 |

---

## 15. 命令速查（复制即用）

```bash
# === 宿主机：转换数据 ===
cd DrivingSDK/model_examples/FlashOCC/FlashOCC
python tools/create_car_perception_flashocc.py --src /data/car_perception_grid
python tools/link_car_perception_media.py
python tools/fix_car_perception_pkl.py
python tools/visualize_car_grid_samples.py

# === 容器内：环境 ===
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /root/miniconda3/etc/profile.d/conda.sh && conda activate torch2.1.0_py38
export TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export TORCH_NPU_LIB=$(python -c "import torch_npu, os; print(os.path.join(os.path.dirname(torch_npu.__file__), 'lib'))")
export PY_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export ASCEND_CUSTOM_OPP_PATH=${PY_SITE}/mx_driving/packages/vendors/customize
export LD_LIBRARY_PATH=${ASCEND_CUSTOM_OPP_PATH}/op_api/lib:${TORCH_LIB}:${TORCH_NPU_LIB}:${LD_LIBRARY_PATH}
cd /workspace/flashocc_cann83/DrivingSDK/model_examples/FlashOCC/FlashOCC
export PYTHONPATH=$(pwd)/projects:$(pwd)/mmdetection3d:$(pwd):$PYTHONPATH

# === 冒烟 ===
bash test/train_car_grid_smoke.sh 1 1

# === 正式 4 卡训练 ===
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
bash tools/dist_train.sh projects/configs/flashocc/flashocc-r50-car-grid.py 4 \
  --work-dir work_dirs/car_grid_v4 \
  --cfg-options data.samples_per_gpu=2 data.workers_per_gpu=2 runner.max_epochs=12

# === 评测 ===
bash test/eval_car_grid.sh work_dirs/car_grid_v4/epoch_12_ema.pth work_dirs/car_grid_test_results
```

---

*文档更新时间：2026-06-16。配置或数据有变更时，请同步更新对应章节。*
