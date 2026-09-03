# 操作手册：实时天气预报推理（端到端）

一条命令完成「下载 ECMWF 数据 → 处理成标准 .nc → GPU 推理 → 可视化 → 结果保存」全流程。本文档面向**自行运行推理**的使用者，说明命令、参数含义和产物位置。

---

## 0. 总体流程

```
ECMWF 开放数据 (IFS 0.25°)  →  下载 fc0 分析场  →  处理成标准 .nc  →  GraphCast GPU 推理  →  可视化
   opendata_download             data_processing        weathernext_forecast       weathernext_forecast
        (stage 1)                  (stage 2)               (stage 3)                (stage 4)
```

- 三阶段严格解耦：每阶段只**读文件 / 写文件**，无共享内存状态，可独立重跑。
- 下载、处理、画图都在登录节点跑（纯 CPU）；**推理是唯一 GPU 步骤**，经 Slurm 提交到 `gpu` 分区。
- 预测结果统一为**通用格式**，写入**外部** `results/` 目录树，按模型组织；默认只保存**中国区**（约 0.4 GB），不再落 13 GB 的全球文件。

---

## 1. 环境准备（一次性）

Setonix 的 conda base 环境损坏，**不要用 `conda activate`**，用 PATH 方式激活推理环境：

```bash
export PATH=/scratch/pawsey0115/hwang4/miniconda3/envs/infer-gpu/bin:$PATH
```

所需依赖已装好：`jax`(ROCm)、`ecmwf-opendata`、`cfgrib`、`netCDF4`、`eccodes`、`cartopy`、`matplotlib`、`Pillow`。

---

## 2. 一键端到端（推荐）

在 `weathernext_forecast/` 目录执行：

```bash
cd /scratch/pawsey0115/hwang4/projects/Foehn/weathernext_forecast
./scripts/realtime.sh                 # 自动取最新一轮分析场，跑完整流程
```

`realtime.sh` 依次：① 解析最新可用分析场时刻 → ② 下载 → ③ 处理 → ④ 提交 GPU 推理并等待（`--wait`）→ ⑤ 画图。

常用变体：

| 命令 | 作用 |
|------|------|
| `./scripts/realtime.sh --no-submit` | 只下载 + 处理，不推理（先检查数据是否正常） |
| `./scripts/realtime.sh --no-visualize` | 推理但不画图 |
| `./scripts/realtime.sh --date 2026-08-31 --time 00` | 指定起报时刻（默认 `--latest` 自动取最新） |
| `./scripts/realtime.sh --force` | 强制重新下载 / 重新处理 |
| `./scripts/realtime.sh --partition gpu --walltime 08:00:00` | 指定分区和更长墙钟 |

---

## 3. 参数速查表（`realtime.sh` 透传给 `realtime.py`）

### 周期选择

| 参数 | 含义 |
|------|------|
| `--date YYYY-MM-DD` | 起报日期（UTC） |
| `--time HH` | 起报时刻，取 `00/06/12/18` 之一（IFS 每天 4 轮分析） |
| `--latest` | 用最新可用分析场（无 `--date/--time` 时默认） |
| `--source` | 数据源：`google`(默认) / `aws` / `azure` / `ecmwf` |

### 下载 / 处理

| 参数 | 含义 |
|------|------|
| `--data-root PATH` | 原始数据根目录，默认 `<repo>/data`（可用环境变量 `OPENDATA_DATA_ROOT` 覆盖） |
| `--force` | 重新下载 / 重新处理（默认跳过已存在文件） |
| `--no-static` | 跳过静态场（地形、海陆掩膜；首次必须下载一次） |

### 推理（Slurm）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--account` | `pawsey0115-gpu` | Slurm 账号 |
| `--partition` | `gpu-dev` | 分区：`gpu` / `gpu-dev` / `gpu-highmem`（长时任务建议用 `gpu`） |
| `--nodes` | `1` | 节点数 |
| `--gpus-per-node` | `3` | GPU 数。**每卡固定 ~28.75 GiB，0.25° 推理峰值 ~61.5 GiB，必须 ≥3 卡** |
| `--walltime` | `04:00:00` | 墙钟限制（首次编译 5–20 分钟；有编译缓存后 ~1–2 分钟） |
| `--no-wait` | — | 提交后立即返回，不阻塞等结果 |
| `--conda-env` | infer-gpu 路径 | 推理环境路径 |

### 可视化

| 参数 | 含义 |
|------|------|
| `--no-visualize` | 跳过画图（可视化固定画 3 城：北京/上海/广州，见 §6） |

---

## 4. 手动分步运行（进阶 / 排错）

各阶段可独立运行，方便定位问题：

```bash
# ① 下载（在 forecast_models/ 仓库根目录）
python opendata_download/scripts/download_fc0.py --date 2026-08-31 --time 00
#   或自动取最新： --latest

# ② 处理成标准 .nc（写入 weathernext_forecast/data/processed/）
python data_processing/scripts/prepare_fc0.py --date 2026-08-31 --time 00

# ③ GPU 推理（在 weathernext_forecast/ 下，先指到刚处理出的文件）
cd weathernext_forecast
export WEATHERNEXT_INPUT_FILENAME=source-ifs_date-2026-08-31_res-0.25_levels-13_steps-40.nc
sbatch --account=pawsey0115-gpu --partition=gpu --nodes=1 --gpus-per-node=3 \
       --time=02:00:00 --chdir=$PWD --wait scripts/run_inference.sbatch

# ④ 画图（读 results/ 树里刚生成的预测，自动取最新）
python scripts/visualize.py --predictions \
  ../results/graphcast/operational/2026-08-31T00Z/predictions/graphcast_operational_IC2026-08-31T00_STEPS40_240h_0.25deg_china.nc
```

> 推理也可以不经过 Slurm，直接在 GPU 节点上 `python scripts/run_inference.py`（先 `export WEATHERNEXT_INPUT_FILENAME=...` 指到输入文件）。

---

## 5. 结果文件（通用预测格式）

预测写入**外部**目录树，按模型组织：

```
<repo>/results/<model>/<variant>/<init>Z/
├── predictions/   <model>_<variant>_IC<init>_STEPS<n>_<horizon>h_<res>deg_<region>.nc
└── visualizations/  见 §6
```

命名规则（以 `graphcast_operational_IC2026-08-27T00_STEPS40_240h_0.25deg_china.nc` 为例）：

| 片段 | 含义 |
|------|------|
| `graphcast` / `operational` | 模型家族 / 变体 |
| `IC2026-08-27T00` | 起报时刻（IC = initial condition） |
| `STEPS40` | 40 个预报步 |
| `240h` | 预报时长 240 小时（10 天，每 6 小时一步） |
| `0.25deg` | 空间分辨率 0.25° |
| `china` | 保存区域（`china` 或 `global`） |

通用格式（`unified-forecast-1` 约定，与具体模型无关）：

- **无 `batch` 维**（单例维度已去掉）；
- `time` 是**绝对有效时间**（`init + lead`），不再是 timedelta；
- 额外坐标：`init_time`（起报时刻）、`lead_time`（距起报的小时数）；
- 全局属性自描述：`model / variant / init_time / resolution_deg / steps / step_h / horizon_h / region / source_file / convention`。

### 区域控制（中国区 vs 全球）

默认只保存**中国区**（15–55°N，70–140°E），约 0.4 GB。用环境变量切换：

```bash
export WEATHERNEXT_PREDICT_REGION=global   # 保存完整全球（约 13 GB，谨慎）
```

（默认 `china`。写全球后可用 `crop_region.py` 事后裁剪，见下。）

### 事后裁剪已保存的预测

```bash
python weathernext_forecast/scripts/crop_region.py \
  --predictions <某全球预测.nc> --region china --out-root results --delete-global
```

---

## 6. 可视化输出

写入 `results/<model>/<variant>/<init>Z/visualizations/`，共 10 个文件：

| 目录 | 文件 | 内容 |
|------|------|------|
| `series/` | `timeseries_Beijing_39.90N_116.40E.png`<br>`timeseries_Shanghai_31.23N_121.47E.png`<br>`timeseries_Guangzhou_23.13N_113.26E.png` | 3 城时序曲线（2×2 面板：2 米气温 / 10 米风速 / 海平面气压 / 6h 降水），横轴为**实际时间戳**（UTC） |
| `overview/` | `2m_temperature_40steps.png`<br>`wind_10m_40steps.png`<br>`wind_100m_40steps.png` | 3 个变量各自 40 步的小多图总览 |
| `gif/` | `anim_2m_temperature.gif`<br>`anim_wind_10m.gif`<br>`anim_wind_100m.gif`<br>`anim_mslp_wind10m.gif` | 4 个动图（40 帧）：2 米气温、10 米风速、100 米风速、海平面气压+10 米风 |

---

## 7. 同步 Pawsey ↔ 本地

在本地（Windows，Git Bash）执行仓库根目录 `forecast_models/` 下的脚本：

```bash
./sync.sh           # 拉取 results/ 树到本地（默认；绝不传 ≥1 GB 的文件）
./sync.sh pull      # 同上
./sync.sh push      # 把本地源码改动推到 Pawsey（scripts + src，绝不传数据/模型/预测）
```

> 中国区预测（~0.4 GB）和可视化图片会传回本地；完整全球预测（~13 GB）被自动排除。

---

## 8. 常见问题

| 现象 | 原因 / 处理 |
|------|-------------|
| `import jax` 卡死 | jaxlib `.so` 损坏，`pip --force-reinstall --no-deps jaxlib==0.4.35 ...` 重装；或冷 GPU，先 `rocminfo` 预热（sbatch 已内置） |
| `[5/6]` 之后长时间无输出 | 首次 ROCm 编译 5–20 分钟属正常；有编译缓存后 ~1 分钟 |
| Slurm 报 `memory ... rejected` | GPU 分区按卡分配内存（28.75 GiB/卡），不能用 `--mem`，需增加 `--gpus-per-node` |
| 0.25° 推理 OOM | 至少 3 卡（86 GiB）；或 `--exclusive` 独占整节点 |
| 下载 `503 Slow Down` | AWS 源限流，默认已用 `google` 源；可换 `--source` |
| 推理后找不到预测 | 预测已改到外部 `results/` 树（非旧的 `predictions/`），见 §5 |
