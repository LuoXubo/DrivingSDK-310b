# car_perception_grid 数据集 FlashOCC 适配说明

本文档记录将自定义数据集 `/data/car_perception_grid` 适配到 DrivingSDK 8.3 + FlashOCC 训练流程的全部改动、数据处理逻辑与验证结果。

---

## 1. 背景与目标

### 1.1 原始需求

在 nuScenes 上验证通 FlashOCC 环境后，使用自有数据进行 OCC 训练，核心约束如下：

| 项目 | nuScenes 默认 | 本数据集目标 |
|------|--------------|-------------|
| 相机数量 | 6 | **2**（左/右前视） |
| OCC 高度层数 Dz | 16 | **2** |
| 语义类别数 | 18 | **3**（可通行 / 车辆障碍 / 未知） |
| 深度监督 | 由点云投影生成 | **直接读取预计算 depth.npy** |

### 1.2 语义类别映射

原始分割图（BGR 的 B 通道）与 FlashOCC 训练标签对应关系：

| 原始 seg 值 | 含义 | OCC 标签 |
|------------|------|---------|
| 0 | 可通行 (passable) | **0** |
| 1 | 车辆障碍 (car) | **1** |
| 2 | 未知 (unknown) | **255**（`ignore_index`，不参与 loss） |

---

## 2. 原始数据格式

### 2.1 目录结构

```
/data/car_perception_grid/
├── near_20260610_224039/          # 约 17 帧
│   └── poses/<pose_name>/
└── wave1_20260610_092058/         # 约 1604 帧
    └── poses/<pose_name>/
```

每个 `poses/<pose_name>/` 下，同一时刻（`stem`）包含：

| 文件模式 | 说明 |
|---------|------|
| `{stem}_cam1.png` / `{stem}_cam2.png` | 左/右相机图像，1024×1024，**灰度 (L 模式)** |
| `{stem}_cam1_depth.npy` / `{stem}_cam2_depth.npy` | 深度图 float32，无效值约 65504 |
| `{stem}_seg_cam1_raw.png` / `{stem}_seg_cam2_raw.png` | 分割图，B 通道编码类别 0/1/2 |
| `{stem}_lidar_sensor.ply` | 雷达/点云（xyz float32） |
| `{stem}_occ.npz`（仅少数帧） | 预计算 16 层 OCC GT |
| `{stem}_gt.json` / `{stem}_ul01_meta.json` | 元数据 |

### 2.2 统计（转换时）

- 有效 RGB 样本：**1621** 帧（两路相机齐全）
- 会话：`near_20260610_224039` + `wave1_20260610_092058`
- 预计算 `*_occ.npz`：仅 **4** 帧，其余由 seg + depth + lidar 在线栅格化生成

### 2.3 OCC 栅格参数（与原始 `*_occ.npz` 一致）

```
X 范围: 0.1 m ~ 9.85 m（车体前方）
Y 范围: -5.05 m ~ 5.0 m（左右）
Z 范围: -2.0 m ~ 1.0 m
分辨率: 0.15 m
原始网格: 67 × 67 × 16
输出网格: 200 × 200 × 2（前 67×67 有效，其余 padding）
Z 两层划分: layer0 = z[0:8)，layer1 = z[8:16)
```

### 2.4 相机标定（moon_stereo）

两路相机共用内参，左右仅平移不同：

```
fx = fy = 610.18, cx = cy = 512
CAM_FRONT_LEFT  sensor2ego_translation: [0, -0.155, 0]
CAM_FRONT_RIGHT sensor2ego_translation: [0, +0.155, 0]
```

---

## 3. 数据转换流程

### 3.1 输出目录

转换结果位于 FlashOCC 工程内：

```
FlashOCC/data/car_perception_grid/nuscenes/
├── bevdetv2-nuscenes_infos_train.pkl   # 1459 样本
├── bevdetv2-nuscenes_infos_val.pkl     # 158 样本
├── samples/
│   ├── CAM_FRONT_LEFT/*.png
│   ├── CAM_FRONT_RIGHT/*.png
│   ├── depth/CAM_FRONT_LEFT/*_depth.npy
│   ├── depth/CAM_FRONT_RIGHT/*_depth.npy
│   └── LIDAR_TOP/*.bin
└── gts/<scene-name>/<token>/labels.npz
```

每个 `labels.npz` 包含：

- `semantics`：`(200, 200, 2)` uint8，取值 0 / 1 / 255
- `mask_camera`：`(200, 200, 2)` uint8，相机可见区域
- `mask_lidar`：`(200, 200, 2)` uint8，点云命中区域

### 3.2 转换脚本

**主脚本**：`tools/create_car_perception_flashocc.py`

```bash
cd FlashOCC
python tools/create_car_perception_flashocc.py --src /data/car_perception_grid
```

主要逻辑：

1. **样本发现** `discover_samples()`：扫描各 session 的 `poses/`，匹配 `*_cam1.png`（排除 `*_seg_cam*` 误匹配）。
2. **GT 生成**：
   - 若存在 `{stem}_occ.npz`：读取 16 层语义，将 class 2 映射为 255，再 **折叠为 2 层**。
   - 否则 `build_gt_from_sensors()`：由 depth + seg + 外参投影到 BEV 栅格，并用 lidar 点云填充 `mask_lidar`。
3. **16→2 层折叠** `collapse_occ_16_to_2()`：每层取 z 区间内 any-visible 语义，car 优先于 passable。
4. **Padding** `pad_occ()`：67³ → 200×200×2，无效区 semantics=255。
5. **划分 train/val**：默认前 10% 为 val，其余为 train（按样本扫描顺序）。

**辅助脚本**：

| 脚本 | 作用 |
|------|------|
| `tools/link_car_perception_media.py` | 将宿主机 `/data/...` 路径的图片/深度 **硬链接** 到 `samples/`，并重写 pkl 为相对路径（Docker 容器内可访问） |
| `tools/fix_car_perception_pkl.py` | 剔除无效样本（`_seg` 误匹配 stem、文件缺失等） |
| `tools/visualize_car_grid_samples.py` | 导出可视化 panel，用于人工检查 |

### 3.3 Docker 挂载说明

容器 `flashocc_drivingsdk_83` 默认**未挂载** `/data/car_perception_grid`。当前方案是在宿主机执行 `link_car_perception_media.py`，把媒体文件硬链到工作区 `samples/` 下，无需在容器内访问原始路径。

若希望容器直接读原始数据，可在 `docker run` 增加：

```bash
-v /data/car_perception_grid:/data/car_perception_grid
```

---

## 4. 代码改动清单

### 4.1 新增文件

| 路径 | 说明 |
|------|------|
| `tools/create_car_perception_flashocc.py` | 数据集转换 |
| `tools/link_car_perception_media.py` | 媒体硬链 + pkl 路径修正 |
| `tools/fix_car_perception_pkl.py` | pkl 清洗 |
| `tools/visualize_car_grid_samples.py` | 训练样本可视化 |
| `tools/train.py` | NPU 训练入口（插件注册 + mx_driving patcher） |
| `projects/configs/flashocc/flashocc-r50-car-grid.py` | 本数据集训练配置 |
| `test/train_car_grid_smoke.sh` | 冒烟训练脚本 |
| `docs/car_perception_grid_适配说明.md` | 本文档 |

### 4.2 修改文件

#### `projects/mmdet3d_plugin/datasets/pipelines/loading.py`

1. **新增 `LoadDepthGTFromFile`**  
   从 pkl 中 `depth_path` 直接加载深度 `.npy`，替换默认的 `PointToMultiViewDepth`（点云投影深度）。无效深度 ≥ 60000 置 0，并 resize 到网络输入尺寸。

2. **新增 / 导出 `LoadOccGTFromFile`**  
   从 `occ_path/labels.npz` 加载 `semantics`、`mask_camera`、`mask_lidar`。

3. **灰度图转 RGB**（`PrepareImageInputs.get_inputs`）  
   原始 `cam1/cam2.png` 为 L 模式，归一化前 `convert('RGB')`，避免 OpenCV normalize 通道数不匹配。

#### `projects/mmdet3d_plugin/datasets/pipelines/__init__.py`

导出 `LoadOccGTFromFile`、`LoadDepthGTFromFile`。

#### `projects/mmdet3d_plugin/` 插件源码

从 `DrivingSDK_wrs` 同步了完整 `mmdet3d_plugin`（主仓库仅有 `__pycache__`）。

### 4.3 训练配置 `flashocc-r50-car-grid.py`

继承 `flashocc-r50-perf.py`，覆盖以下关键项：

```python
# 数据
data_root = 'data/car_perception_grid/nuscenes/'
cams = ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
Ncams = 2
src_size = (1024, 1024)

# 管线：去掉 PointToMultiViewDepth，改用 LoadDepthGTFromFile
train_pipeline = [
    PrepareImageInputs,
    LoadAnnotationsBEVDepth,   # BDA 增强关闭（rot/scale/flip=恒等）
    LoadOccGTFromFile,
    LoadDepthGTFromFile,
    ...
]

# 模型
model.occ_head.Dz = 2
model.occ_head.num_classes = 3
model.occ_head.loss_occ.ignore_index = 255
```

预训练权重仍使用基类配置：`ckpts/bevdet-r50-cbgs.pth`。

### 4.4 NPU 适配（训练前执行）

对 `bevdet_occ.py` 做 idempotent sed（`test/train_car_grid_smoke.sh` 内）：

- 注释 `ThreadPool`、`nearest_assign` 导入
- `is_cuda = True` → `is_cuda = False`

### 4.5 `tools/train.py`（新增）

官方 `flashocc.patch` 期望在 FlashOCC 根目录有 `tools/train.py`，原仓库缺失。新增版本包含：

- `import projects.mmdet3d_plugin` 插件注册（读取 config 中 `plugin=True`）
- `torch_npu` + `mx_driving.patcher`（batch_matmul、resnet_add_relu）
- 解决 `BEVDetOCC is not in the models registry` 错误

---

## 5. 训练与验证

### 5.1 环境（Docker 内）

```bash
docker exec -it flashocc_drivingsdk_83 /bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate torch2.1.0_py38

export ASCEND_RT_VISIBLE_DEVICES=0
export TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export TORCH_NPU_LIB=$(python -c "import torch_npu, os; print(os.path.join(os.path.dirname(torch_npu.__file__), 'lib'))")
export PY_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export ASCEND_CUSTOM_OPP_PATH=${PY_SITE}/mx_driving/packages/vendors/customize
export LD_LIBRARY_PATH=${ASCEND_CUSTOM_OPP_PATH}/op_api/lib:${TORCH_LIB}:${TORCH_NPU_LIB}:${LD_LIBRARY_PATH}
```

### 5.2 冒烟训练

```bash
cd /workspace/flashocc_cann83/DrivingSDK/model_examples/FlashOCC/FlashOCC
bash test/train_car_grid_smoke.sh 1 1
```

已在容器内验证通过（2 样本、1 NPU、1 epoch）：

```
Epoch [1][1/2]  loss_occ: 0.8630
Epoch [1][2/2]  loss_occ: 0.8623
Saving checkpoint at 1 epochs
```

输出：`work_dirs/car_grid_smoke/`

### 5.3 正式训练（示例）

```bash
export PYTHONPATH=$(pwd)/projects:$(pwd)/mmdetection3d:$(pwd):$PYTHONPATH

python tools/train.py projects/configs/flashocc/flashocc-r50-car-grid.py \
  --work-dir work_dirs/car_grid_full \
  --gpu-id 0 \
  --cfg-options data.samples_per_gpu=2 runner.max_epochs=24 data.workers_per_gpu=2
```

---

## 6. 可视化检查

### 6.1 生成可视化

```bash
cd FlashOCC
python tools/visualize_car_grid_samples.py
```

### 6.2 输出位置

```
data/car_perception_grid/vis_train_samples/
├── summary_grid.png      # 多样本总览
├── legend.png            # 颜色图例
├── manifest.json         # 每样本统计
└── <split>_<idx>_<tag>/panel.png
```

每张 panel 包含：双相机图、深度图、分割图、OCC 两层 BEV、lidar mask。

图例：

- 绿 = passable (0)
- 红 = car (1)
- 灰 = ignore (255)
- 橙色叠加 = camera 可见 mask

---

## 7. 已知问题与数据统计

### 7.1 训练集类别不平衡（重要）

| 划分 | 样本数 | 含 car(1) 的样本 | 语义唯一值 |
|------|--------|-----------------|-----------|
| train | 1459 | **0** | [0, 255] |
| val | 158 | **4** | [0, 1, 255] |

**结论**：当前 train 集完全没有车辆障碍标签，car 类仅出现在 val 的 4 帧（`d10p0_az000_cy000_sun0*` 系列）。若要做 3 类 OCC 训练，需要：

- 重新划分 train/val（保证 train 含 car 样本），或
- 补充含障碍物的数据，或
- 调整 `build_gt_from_sensors` 的栅格化逻辑并复核 seg 标注

### 7.2 camera mask 稀疏

训练样本仅约 **3.6%** 体素有 camera 标注，绝大部分为 ignore(255)。这与深度+seg 投影步长（`step=4`）及视野有关，属于当前栅格化策略的结果。

### 7.3 lidar mask

- train 中约 **1368/1459** 帧有非零 `mask_lidar`
- val 中含 car 的 4 帧 lidar mask 为 0（这些帧未投影到点云）

### 7.4 历史 bug 修复记录

| 问题 | 处理 |
|------|------|
| `discover_samples` 将 `*_seg_cam1.png` 误识别为相机图 | glob 时排除 `_seg_cam` |
| pkl 指向宿主机绝对路径，Docker 内找不到文件 | `link_car_perception_media.py` 硬链 + 相对路径 |
| 灰度图导致 `cv2.subtract` 失败 | `loading.py` 转 RGB |
| `BEVDetOCC` 未注册 | 新增 `tools/train.py` 导入插件 |
| mmcv 源码损坏 | 容器内 `pip install mmcv-full==1.7.2` |

---

## 8. 典型命令速查

```bash
# 1. 转换数据集（宿主机，需能访问 /data/car_perception_grid）
python tools/create_car_perception_flashocc.py --src /data/car_perception_grid

# 2. 硬链媒体到工作区（Docker 训练前）
python tools/link_car_perception_media.py

# 3. 清洗 pkl
python tools/fix_car_perception_pkl.py

# 4. 可视化检查
python tools/visualize_car_grid_samples.py

# 5. 冒烟训练（Docker 内）
bash test/train_car_grid_smoke.sh 1 1

# 6. 指定样本可视化
python tools/visualize_car_grid_samples.py --indices 0,729,1458 --max-samples 3
```

---

## 9. 文件路径索引

| 用途 | 路径 |
|------|------|
| 原始数据 | `/data/car_perception_grid` |
| 转换后数据 | `FlashOCC/data/car_perception_grid/nuscenes/` |
| 训练配置 | `FlashOCC/projects/configs/flashocc/flashocc-r50-car-grid.py` |
| 可视化结果 | `FlashOCC/data/car_perception_grid/vis_train_samples/` |
| 预训练权重 | `FlashOCC/ckpts/bevdet-r50-cbgs.pth` |
| 冒烟日志 | `FlashOCC/test/output/car_grid_smoke/train.log` |
| 工作区（容器） | `/workspace/flashocc_cann83/DrivingSDK/model_examples/FlashOCC/FlashOCC` |

---

*文档生成时间：2026-06-11。如有数据或配置更新，请同步修改本文档对应章节。*
