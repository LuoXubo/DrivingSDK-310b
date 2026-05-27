# FlashOCC for PyTorch

## 目录

- [简介](#简介)
    - [模型介绍](#模型介绍)
    - [支持任务列表](#支持任务列表)
    - [代码实现](#代码实现)
- [FlashOCC](#flashocc)
    - [准备训练环境](#准备训练环境)
    - [快速开始](#快速开始)
       - [训练任务](#训练任务) 
       - [Ascend 310B推理验证](#ascend-310b推理验证)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

FlashOCC是一种高效且轻量化的占用预测框架，专为自动驾驶系统中的3D场景理解设计。与现有体素级别方法不同，FlashOCC在BEV（鸟瞰图）空间中保留特征，利用高效的2D卷积进行特征提取，并通过通道到高度的变换将BEV输出提升至3D空间。这一设计显著降低了内存和计算开销，同时保持了高精度。

## 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| FlashOCC |   训练   |    ✔     |
| FlashOCC | Ascend 310B单帧推理/冒烟验证 | ✔ |

> 说明：310B适配目标是FlashOCC推理forward链路，重点覆盖`projects/configs/flashocc/flashocc-r50.py`和`BEVPoolV3` forward。训练、`BEVPoolV3Grad`以及完整性能优化不在本适配范围内。

## 代码实现

- 参考实现：

    ```shell
    url=https://github.com/Yzichen/FlashOCC
    commit_id=4084861d8d605bb01df55fcbc8072036055aa625
    ```

# FlashOCC

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 首次支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 7.0.RC1  |
|       CANN        | 8.1.RC1  |

### 安装模型环境

**表 2**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |   2.1.0   |

0. 激活 CANN 环境（例如：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`）

1. 准备模型源码及安装基础依赖

    在当前目录下，克隆并准备 FlashOCC 源码

    ```shell
    git clone https://github.com/Yzichen/FlashOCC.git
    cp flashocc.patch FlashOCC
    cp -r test/ FlashOCC/
    cd FlashOCC
    git checkout 4084861d8d605bb01df55fcbc8072036055aa625
    git apply --reject --whitespace=fix flashocc.patch
    pip install -r requirements/runtime.txt
    cd ../
    ```

2. 源码编译安装 mmcv

    克隆 mmcv 仓，并进入 mmcv 目录编译安装

    ```shell
    git clone -b 1.x https://github.com/open-mmlab/mmcv
    cp mmcv.patch mmcv
    cd mmcv
    git apply --reject mmcv.patch
    MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
    cd ../
    ```

3. 安装 mmdet

    克隆 mmdet 仓，并进入 mmdet 目录编译安装

    ```shell
    git clone -b v2.25.0 https://github.com/open-mmlab/mmdetection.git
    cp mmdet.patch mmdetection
    cd mmdetection
    git apply --reject mmdet.patch
    pip install -e .
    cd ../
    ```

4. 安装 mmdet3d

    克隆 mmdet3d 仓，并进入 mmdet3d 目录编译安装

    ```shell
    cd FlashOCC
    git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
    cp ../mmdet3d.patch mmdetection3d
    cd mmdetection3d
    git apply --reject mmdet3d.patch
    pip install -v -e .
    cd ../
    ```

5. 安装 Driving SDK 加速库

    安装方法参考[原仓](https://gitcode.com/Ascend/DrivingSDK)

### Ascend 310B推理环境补充

本分支在原有910B实现基础上增加了310B推理适配，核心变化包括：

- `BEVPoolV3` forward新增`ascend310b`注册及kernel分支，310B路径避免使用310B不支持的`Broadcast/BroadCast`。
- Python/C++调用接口保持不变，仍使用`bev_pool_v3(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape)`。
- `BEVPoolV3Grad`未适配310B，310B上仅建议使用`model.eval()`和`torch.no_grad()`进行推理。
- `resnet_add_relu` patch可启用；当前310B环境中`npu_add_relu`走`torch.relu(x + y)` fallback，不依赖`AddRelu`自定义算子。

以下命令以源码位于`/root/flashocc_dir/DrivingSDK`为例。请根据实际路径替换。

1. 激活conda和CANN环境

    ```shell
    source /usr/local/miniconda3/etc/profile.d/conda.sh
    conda activate torch2.1.0_py38

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export ASCEND_CUSTOM_OPP_PATH=$ASCEND_OPP_PATH/vendors/customize:$ASCEND_CUSTOM_OPP_PATH
    export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/customize/op_api/lib:$CONDA_PREFIX/lib/python3.8/site-packages/torch_npu/lib:$LD_LIBRARY_PATH
    export ASCEND_LAUNCH_BLOCKING=1
    ```

    若CANN安装路径不同，可先查找：

    ```shell
    find /usr/local/Ascend /data/Ascend* -name set_env.sh 2>/dev/null
    ```

2. 检查关键Python包

    ```shell
    python - <<'PY'
    mods = ["torch", "torch_npu", "torchvision", "mmcv", "mmdet", "mmdet3d", "mx_driving"]
    for m in mods:
        try:
            mod = __import__(m)
            print(m, "OK", getattr(mod, "__version__", ""), getattr(mod, "__file__", ""))
        except Exception as e:
            print(m, "FAIL", repr(e))
    PY
    ```

    推荐版本组合：

    | 组件 | 已验证版本 |
    | ---- | ---------- |
    | Python | 3.8 |
    | PyTorch | 2.1.0 |
    | torch_npu | 2.1.0.post17 |
    | torchvision | 0.16.0 |
    | mmcv | 1.7.2 |
    | mmdet | 2.25.0 |
    | mmdet3d | 1.0.0rc4 |
    | numpy | 1.23.5 |
    | networkx | 2.2 |
    | trimesh | 2.35.39 |

    OpenMMLab相关源码包建议使用`--no-deps`安装，避免pip自动拉取过新的依赖。

3. 构建并安装310B自定义算子包

    若`mx_driving/packages`下已经存在`ascend310b/bev_pool_v3/BEVPoolV3_*.o`，可直接安装：

    ```shell
    cd /root/flashocc_dir/DrivingSDK/mx_driving
    bash ../scripts/install_kernel.sh --quiet

    find $ASCEND_OPP_PATH/vendors/customize -path '*ascend310b*bev_pool_v3*'
    ```

    若需要从源码重新构建`BEVPoolV3` 310B算子：

    ```shell
    cd /root/flashocc_dir/DrivingSDK

    cmake . --preset=default -B build_310b \
      -DBUILD_STAGE=0 \
      -DMX_DRIVING_PATH=$PWD/mx_driving \
      -DASCEND_COMPUTE_UNIT=ascend310b \
      -DKERNEL_NAME=BEVPoolV3 \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build build_310b -j$(nproc)

    cmake . --preset=default -B build_310b \
      -DBUILD_STAGE=1 \
      -DMX_DRIVING_PATH=$PWD/mx_driving \
      -DASCEND_COMPUTE_UNIT=ascend310b \
      -DKERNEL_NAME=BEVPoolV3 \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build build_310b -j$(nproc)

    cd mx_driving
    bash ../scripts/install_kernel.sh --quiet
    ```

4. 可选：单算子forward正确性检查

    ```shell
    cd /root/flashocc_dir/DrivingSDK
    python - <<'PY'
    import torch
    import torch_npu
    from mx_driving import bev_pool_v3

    def golden(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape):
        B, D, H, W, C = bev_feat_shape
        depth = depth.view(-1)
        feat = feat.view(-1, C)
        weighted = depth[ranks_depth.long()].unsqueeze(1) * feat[ranks_feat.long()]
        out = torch.zeros(B * D * H * W, C, device=depth.device)
        out.index_add_(0, ranks_bev.long(), weighted)
        return out.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

    B, D, H, W, C, N = 1, 5, 17, 23, 8, 777
    depth = torch.rand([B, 1, D, H, W])
    feat = torch.rand([B, 1, H, W, C])
    ranks_depth = torch.randint(0, B * D * H * W, [N], dtype=torch.int32)
    ranks_feat = torch.randint(0, B * H * W, [N], dtype=torch.int32)
    ranks_bev = torch.randint(0, B * D * H * W, [N], dtype=torch.int32)
    shape = [B, D, H, W, C]

    ref = golden(depth, feat, ranks_depth, ranks_feat, ranks_bev, shape)
    out = bev_pool_v3(depth.npu(), feat.npu(), ranks_depth.npu(),
                      ranks_feat.npu(), ranks_bev.npu(), shape).cpu()
    print("max_abs_err =", (out - ref).abs().max().item())
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4)
    print("BEVPoolV3 310B forward OK")
    PY
    ```

### 准备数据集

1. 根据原仓[Environment Setup](https://github.com/Yzichen/FlashOCC/blob/master/doc/install.md) 在模型源码根目录下准备数据集，参考数据集结构如下：

    ```shell
    └── Path_to_FlashOcc/
    └── data
        └── nuscenes
            ├── v1.0-trainval
            ├── maps
            ├── panoptic
            ├── lidarseg
            ├── sweeps
            ├── samples
            ├── gts
            ├── bevdetv2-nuscenes_infos_train.pkl (经数据预处理后生成)
            └── bevdetv2-nuscenes_infos_val.pkl (经数据预处理后生成)
    ```

2. 在模型源码根目录下进行数据预处理

   ```shell
   python tools/create_data_bevdet.py
   ```

### 准备预训练权重

在模型源码根目录下创建 ckpts 文件夹，将预训练权重 [bevdet-r50-cbgs.pth](https://drive.usercontent.google.com/download?id=1oWkQLmzAXi_AoJZ259EbRmksbOyBbYuX&export=download&authuser=0) 放入其中

   ```shell
   ckpts/
   ├── bevdet-r50-cbgs.pth
   ```

## 快速开始

### 训练任务

本任务主要提供**单机**的**8卡**训练脚本。

#### 开始训练

  - 在模型源码根目录下，运行训练脚本。
    
    运行脚本支持命令行参数：
    - '--num-npu'：NPU卡数，默认为8；
    - '--batch-size': 每卡batch-size大小，默认为24；
    - 单机8卡性能训练

     ```shell
     bash test/train_8p_flashocc_r50_perf.sh
     (option) bash test/train_8p_flashocc_r50_perf.sh --num-npu 8 --batch-size 24 # 8卡性能
     ```

     - 单机8卡精度训练

     ```shell
     bash test/train_8p_flashocc_r50_full.sh
     (option) bash test/train_8p_flashocc_r50_full.sh --num-npu 8 --batch-size 24 # 8卡精度
     ```

     - 单机8卡backbone FP16性能训练

     ```shell
     bash test/train_8p_flashocc_r50_fp16_backbone_perf.sh
     (option) bash test/train_8p_flashocc_r50_fp16_backbone_perf.sh --num-npu 8 --batch-size 24
     ```

     - 单机8卡backbone FP16精度训练

     ```shell
     bash test/train_8p_flashocc_r50_fp16_backbone_full.sh
     (option) bash test/train_8p_flashocc_r50_fp16_backbone_full.sh --num-npu 8 --batch-size 24
     ```

#### 训练结果

| 芯片          | 卡数 | global batch size | Precision | epoch | mIoU | 性能-单步迭代耗时(s) | FPS |
| ------------- | :--: | :---------------: | :-------: | :---: | :----: |  :-------------------: |  :-----------------:   |
| 竞品A           |  8p  |         192         |   fp32    |  24   | 30.14 |        2.83          |   67.98    |
| Atlas 800T A2 |  8p  |         192         |   fp32    |  24   | 30.27 |          1.83          |   104.85   |

#### backbone FP16训练结果

| 芯片          | 卡数 | global batch size | Precision | epoch | mIoU | 性能-单步迭代耗时(s) | FPS |
| ------------- | :--: | :---------------: | :-------: | :---: | :----: |  :-------------------: |  :-----------------:   |
| Atlas 800T A2 |  8p  |         192         |   backbone fp16    |  24   | 30.15 |           1.34          |   143.17   |

### Ascend 310B推理验证

#### 冒烟测试

冒烟测试用于验证数据加载、模型构建、checkpoint加载、`BEVPoolV3` forward和mIoU评估链路。建议先使用少量val样本，例如10帧smoke数据。

1. 准备最小数据目录

    在FlashOCC源码根目录下，配置应能访问：

    ```shell
    data/nuscenes/
    ├── bevdetv2-nuscenes_infos_val.pkl
    ├── samples/    # 只需要val pkl实际引用的图片和LIDAR_TOP bin
    └── gts/        # 只需要val pkl实际引用的labels.npz
    ```

    若数据放在其他目录，可建立软链接：

    ```shell
    cd /root/flashocc_dir/DrivingSDK/model_examples/FlashOCC/FlashOCC
    mkdir -p data
    ln -sfn /root/flashocc_dir/flashocc_data/nuscenes data/nuscenes
    ```

2. 降低DataLoader并发，避免310B小内存场景卡住

    ```shell
    cd /root/flashocc_dir/DrivingSDK/model_examples/FlashOCC/FlashOCC
    python - <<'PY'
    from pathlib import Path
    p = Path("projects/configs/flashocc/flashocc-r50.py")
    s = p.read_text()
    s = s.replace("samples_per_gpu=24", "samples_per_gpu=1")
    s = s.replace("workers_per_gpu=24", "workers_per_gpu=1")
    p.write_text(s)
    PY
    ```

3. 执行smoke推理

    ```shell
    source /usr/local/miniconda3/etc/profile.d/conda.sh
    conda activate torch2.1.0_py38
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export ASCEND_CUSTOM_OPP_PATH=$ASCEND_OPP_PATH/vendors/customize:$ASCEND_CUSTOM_OPP_PATH
    export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/customize/op_api/lib:$CONDA_PREFIX/lib/python3.8/site-packages/torch_npu/lib:$LD_LIBRARY_PATH
    export ASCEND_LAUNCH_BLOCKING=1

    cd /root/flashocc_dir/DrivingSDK/model_examples/FlashOCC/FlashOCC
    timeout -k 30s 45m python tools/test.py \
      projects/configs/flashocc/flashocc-r50.py \
      /root/flashocc_dir/epoch_24_ema.pth \
      --eval mIoU 2>&1 | tee flashocc_310b_smoke45m.log
    ```

    10帧smoke数据跑通时，日志中会出现类似输出：

    ```text
    [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 10/10
    metric =  mIoU
    ===> mIoU of 10 samples: 32.8
    ```

    首次运行可能触发NPU算子编译/缓存，第一帧会明显慢于后续运行。若只想限制验证时长，可调整`timeout`，例如`timeout -k 30s 5m ...`。

#### 假数据单帧测试

`tools/fake_frame_infer_310b.py`可直接构造一帧完整的假输入，不依赖NuScenes数据文件，适合模拟服务化场景中的“单帧输入->FlashOCC forward->occupancy输出”链路。

输入张量结构：

```text
imgs:         (1, 6, 3, 256, 704)
sensor2egos:  (1, 6, 4, 4)
ego2globals:  (1, 6, 4, 4)
intrins:      (1, 6, 3, 3)
post_rots:    (1, 6, 3, 3)
post_trans:   (1, 6, 3)
bda:          (1, 3, 3)
```

运行命令：

```shell
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate torch2.1.0_py38
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=$ASCEND_OPP_PATH/vendors/customize:$ASCEND_CUSTOM_OPP_PATH
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/customize/op_api/lib:$CONDA_PREFIX/lib/python3.8/site-packages/torch_npu/lib:$LD_LIBRARY_PATH
export ASCEND_LAUNCH_BLOCKING=1

cd /root/flashocc_dir/DrivingSDK/model_examples/FlashOCC/FlashOCC
timeout -k 30s 45m python tools/fake_frame_infer_310b.py \
  --config projects/configs/flashocc/flashocc-r50.py \
  --checkpoint /root/flashocc_dir/epoch_24_ema.pth \
  --output /root/flashocc_dir/fake_frame_occ_output.npz \
  --image-mode random
```

成功时会打印：

```text
fake imgs shape: (1, 6, 3, 256, 704)
first output shape: (200, 200, 16)
first output dtype: uint8
saved: /root/flashocc_dir/fake_frame_occ_output.npz
```

如需查看耗时分解，可增加`--profile`：

```shell
timeout -k 30s 45m python tools/fake_frame_infer_310b.py \
  --config projects/configs/flashocc/flashocc-r50.py \
  --checkpoint /root/flashocc_dir/epoch_24_ema.pth \
  --output /root/flashocc_dir/fake_frame_occ_output_profile.npz \
  --image-mode random \
  --profile 2>&1 | tee fake_frame_profile_310b.log
```

已验证的一次热启动profiling结果如下，仅作为310B PyTorch eager路径参考：

```text
prepare_inputs             44.598s
img_backbone               74.651s
img_neck                    2.111s
view_transformer_bevpool   24.534s
bev_encoder_backbone        7.208s
bev_encoder_neck           22.299s
occ_head_forward           13.053s
occ_head_get_occ_cpu        1.719s
profiled total            190.173s
```

实际部署时建议将模型做成常驻进程：启动时完成`build_model`、`load_checkpoint`、`model.npu()`和warmup；每帧只执行输入准备与forward，避免重复支付初始化开销。

#### 已知限制

- 310B目前仅验证FlashOCC R50单帧推理forward链路。
- `BEVPoolV3Grad`未适配310B，训练或显式backward会失败。
- `AddRelu`自定义算子未提供310B kernel；当前`resnet_add_relu` patch在310B上通过`torch.relu(x + y)` fallback保证可用性。
- 该路径为PyTorch eager推理验证，不代表最终部署性能。首次运行通常会因NPU算子编译和缓存生成而显著变慢。

# 变更说明

2026.5.27：新增Ascend 310B推理适配说明，补充BEVPoolV3 310B构建、冒烟测试和假数据单帧测试流程。

2025.3.13：首次发布。

2025.4.28：性能优化。

2025.7.23：优化fps计算方式，添加backbone fp16混合精度训练。

2025.8.20：增大num worker，更新fp16性能。

2025.8.25：优化训练脚本，增加入参。

# FAQ

## 训练时报错`ImportError: cannot import name 'gcd' from 'fractions'` 

报错原因为networkx版本低，使用`pip install networkx==3.1`升级依赖版本即可。

## 310B推理时报错`module 'numpy' has no attribute 'int'`

310B推理环境建议保持`numpy==1.23.5`、`networkx==2.2`和`trimesh==2.35.39`的组合。若`numpy>=1.24`搭配旧版`networkx`，可能在导入`trimesh`或`mmdet3d`时触发`np.int`错误。可执行：

```shell
python -m pip install "numpy==1.23.5" "networkx>=2.2,<2.3" "trimesh>=2.35.39,<2.35.40"
```

## 训练时报错`libGL.so.1`文件不存在

使用opencv-python时，需配套安装相同版本的opencv-python-headless，使用opencv-contrib-python时，需配套安装相同版本的opencv-contrib-python-headless依赖。
