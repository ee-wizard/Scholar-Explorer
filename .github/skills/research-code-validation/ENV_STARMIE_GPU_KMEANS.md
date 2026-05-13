# Starmie 环境配置备忘（uv + CuPy KMeans）

> 更新时间：2026-05-12
> 适用范围：`/home/wizard/projects/GeneralExplorer/projects/starmie`

## 1) 路径与环境基线

- 工作区根目录：`/home/wizard/projects/GeneralExplorer`
- 项目目录：`/home/wizard/projects/GeneralExplorer/projects/starmie`
- 虚拟环境：`/home/wizard/projects/GeneralExplorer/.venv-unix`
- Python：3.12.13（`uv` 管理）

## 2) 已确认的软件栈

- RSQ KMeans 后端：**CuPy 优先，NumPy 自动兜底**（已在 `starmie_rsq_utils.py` 实装）
- 不再使用 FAISS 做 KMeans（检索逻辑仍可独立存在）

## 3) 依赖安装顺序（推荐）

在激活 `.venv-unix` 后安装：

1. 基础依赖
   - `numpy`
   - `cupy-cuda12x`

2. CuPy 常见缺失库（按报错补齐）
   - 缺 `libcurand.so` → `nvidia-curand-cu12`
   - 缺 `libnvrtc.so` → `nvidia-cuda-nvrtc-cu12`
   - 缺 `libcublasLt.so` → `nvidia-cublas-cu12`

3. 项目运行常见依赖
   - `mlflow`
   - `hnswlib`
   - `munkres`
   - `tqdm`

## 4) uv/PEP668 说明

当调用的是 uv 管理解释器的全局 pip（非项目 venv 内 pip）时，可能出现：

- `externally-managed-environment`

此时需使用 `--break-system-packages`。优先策略：

- 尽量在 `.venv-unix` 内使用本地 `pip`
- 仅在必要时使用 `--break-system-packages`

## 5) CUDA 路径手动指定（CuPy 自动探测失败时）

当出现 `Failed to auto-detect CUDA root directory`，可显式设置：

- `CUDA_PATH=<venv>/lib/python3.12/site-packages/nvidia`
- `LD_LIBRARY_PATH` 追加：
  - `<venv>/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib`
  - `<venv>/lib/python3.12/site-packages/nvidia/curand/lib`
  - `<venv>/lib/python3.12/site-packages/nvidia/cublas/lib`（若已安装 cublas）

## 6) 最小验证清单

1. 导入验证：`numpy`、`cupy` 可导入
2. CUDA 可见：`cupy.cuda.is_available() == True`
3. 在 `projects/starmie` 目录调用 `build_rsq_artifacts`
4. 若 GPU 库不全，日志应显示回退：`falling back to NumPy KMeans`

## 7) 已知坑位

- 在工作区根目录直接 `python - <<PY ... from starmie_rsq_utils import ...` 会因为模块路径不在 `PYTHONPATH` 报错；应在 `projects/starmie` 目录运行。
- `nvidia-cublas-cu12` 体积较大（数百 MB），安装耗时长属正常。
- 即使 `cupy.cuda.is_available() == True`，仍可能因单个动态库缺失而回退 CPU。

## 8) 数据集路径约定（本项目）

- 统一使用：`/mnt/f/Datasets/TableDiscovery/`
- 禁止继续使用：`/f/Datasets/TableDiscovery/`
- 若脚本存在 `data/...` 硬编码，优先修改代码加载层适配真实路径，不改动数据集目录结构。
