# 实时天气预报（双模型）—— 统一操作 / 命令提交指南

覆盖 **GraphCast (WeatherNext 1 Graph)** 与 **Aurora 0.25° fine-tuned** 两个模型的完整实时预报流程。本文件是**仓库顶层的外部总入口**，回答一个问题：*怎么提交一次完整预报*。

各模型内部的变量契约、权重下载、排错细节见各自的手册：

- [`weathernext_forecast/OPERATION.md`](weathernext_forecast/OPERATION.md) — GraphCast
- [`aurora_forecast/OPERATION.md`](aurora_forecast/OPERATION.md) — Aurora

---

## 0. 系统总览

一条命令跑完两个模型、**同一分析时刻**的完整链路：

```
ECMWF 开放数据 (IFS 0.25°)
   └─ 下载  opendata_download（共享，模型无关）
        ├─ 处理 → GraphCast 输入  data_processing      ─┐
        │                                              ├─ Slurm GPU 推理 ─┐
        └─ 适配 → Aurora 输入   aurora_forecast.adapter ─┘                  │
                                                                           ▼
                                        results/<model>/<variant>/<init>Z/
                                          ├─ predictions/    统一格式 .nc
                                          └─ visualizations/ {series, overview, gif}
```

- **下载 / 处理 / 适配 / 画图**：登录节点纯 CPU。
- **GPU 推理**：唯一的 GPU 步骤，经 Slurm 提交（`--wait` 等待完成）。
- 预测统一写入**外部** `results/` 树，按模型组织；默认只保存**中国区**（约 0.4 GB），不落 13 GB 的全球文件。
- 顶层入口只查询一次 `--latest`，然后把同一组 `--date/--time` 传给两个模型，避免跨周期边界时起报时刻不一致。

### 目录职责

```text
Foehn/
├── scripts/                 顶层控制脚本（只做跨模型编排）
├── foehn_core/              模型无关的结果契约与原子落盘
├── opendata_download/       ECMWF 原始数据下载
├── data_processing/         GraphCast 输入预处理
├── weathernext_forecast/    GraphCast 模型适配、推理与可视化
├── aurora_forecast/         Aurora 模型适配与推理
├── data/                    共享原始数据（运行产物，不入 Git）
├── results/                 统一预测与图片（运行产物，不入 Git）
└── logs/                    顶层运行日志（运行产物，不入 Git）
```

依赖方向固定为 `模型包 -> foehn_core`；Aurora 不再反向依赖 GraphCast 包。模型权重只存放在各模型目录，顶层不保留重复权重目录。

---

## 1. 一键提交（推荐，双模型）

在仓库根目录（Pawsey 上为 `Foehn/`，本地为 `forecast_models/`）：

```bash
./realtime_all.sh --partition gpu-dev
```

依次执行 GraphCast → Aurora（各自的下载/处理/推理/可视化）。共享的下载阶段会跑两遍，第二次是幂等的快速 no-op。
顶层会把完整控制台输出写入 `logs/realtime_all_<UTC时间>.log`；任一阶段失败时退出码非零并在日志末尾记录失败行。

```bash
# 指定起报时刻
./realtime_all.sh --date 2026-09-02 --time 18 --partition gpu-dev

# 显式取最新（无参数时默认即最新）
./realtime_all.sh --latest --partition gpu-dev
```

---

## 2. 单模型提交

```bash
# GraphCast 单独跑
./weathernext_forecast/scripts/realtime.sh --partition gpu-dev

# Aurora 单独跑
./aurora_forecast/scripts/realtime.sh --partition gpu-dev
```

> 每个 `realtime.sh` 会自己设置 conda 环境的 PATH（Setonix 的 base 环境损坏，**不要用 `conda activate`**），无需手动 `export`。

---

## 3. 关键参数（两模型通用，透传给 `realtime.py`）

### 周期选择

| 参数 | 含义 |
|------|------|
| `--latest` | 用最新可用分析场（无 `--date/--time` 时默认） |
| `--date YYYY-MM-DD` | 起报日期（UTC） |
| `--time HH` | 起报时刻，`00/06/12/18` 之一（IFS 每天 4 轮分析） |
| `--source` | 数据源：`google`（默认）/ `aws` / `azure` / `ecmwf` |

### GPU 推理（Slurm）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--partition` | `gpu` | **当前必须显式传 `gpu-dev`**（`gpu` 分区长期 drain/占满，见 §6） |
| `--account` | `pawsey0115-gpu` | Slurm 账号 |
| `--nodes` | `1` | 节点数 |
| `--gpus-per-node` | GraphCast 3 / Aurora 1 | GCD 数；每 GCD 固定 ~28.75 GiB，GraphCast 0.25° 推理峰值 ~61.5 GiB，需 ≥3 |
| `--walltime` | `04:00:00` / `02:00:00` | 墙钟（首次编译 5–20 分钟；有缓存后 ~1 分钟） |
| `--no-submit` | — | 只下载 + 处理/适配，不推理 |
| `--no-wait` | — | 提交后立即返回（不等待、不自动可视化） |

### 其他

| 参数 | 说明 |
|------|------|
| `--force` | 强制重新下载 / 重新处理 |
| `--no-static` | 跳过静态场（首次必须下载一次） |
| `--no-visualize` | 跳过画图（可视化固定画北京/上海/广州三城） |

---

## 4. 结果

```
results/<model>/<variant>/<init>Z/
├── predictions/    <model>_<variant>_IC<init>_STEPS40_240h_0.25deg_china.nc
└── visualizations/
    ├── series/     3 城时序（北京/上海/广州）：2m 气温 / 10m 风速 / 气压 / 降水
    ├── overview/   3 变量 × 40 步小多图（2m 气温 / 10m 风速 / 100m 风速）
    └── gif/        4 个动图（气温 / 10m 风 / 100m 风 / MSLP+10m 风）
```

- `<model>/<variant>`：`graphcast/operational`、`aurora/0.25-finetuned`。
- 统一格式（`unified-forecast-1`）：无 `batch` 维、`time` 为绝对有效时间、全局属性自描述。
- 每张图的标题都带**模型全称**：
  - `GraphCast (WeatherNext 1 Graph, operational 0.25°)`
  - `Aurora 0.25° fine-tuned (IFS HRES T0)`
- Aurora `0.25-finetuned` 不输出降水，其时序图第 4 面板会明确标注「该模型未预测降水」，不会伪造。

---

## 5. 同步（Pawsey ↔ 本地）

在本地（Windows，Git Bash）仓库根目录执行：

```bash
./sync.sh           # 拉 results/ 树到本地（默认；绝不传 ≥1 GB 文件）
./sync.sh pull      # 同上
./sync.sh push      # 推源码到 Pawsey（scripts + src；绝不传数据/模型/预测）
```

同步白名单包括顶层入口、公共包、四个业务包的 `src/`、`scripts/`、打包元数据和操作文档。`pull` 只合并 `results/`，不会删除本地已有结果；任何单文件达到或超过 1 GiB 都不会传输。

---

## 6. 分区与资源（重要）

- **分区**：`gpu` 分区当前大量节点 drain/被占用，**提交务必用 `--partition gpu-dev`**。
- **资源**：GraphCast 需 3 GCD（~86 GiB）；Aurora 只需 1 GCD（MI250X 双 GCD，gfx90a 原生支持）。
- **环境**（两模型不同）：
  - GraphCast：`/scratch/pawsey0115/hwang4/miniconda3/envs/infer-gpu`（JAX/ROCm）
  - Aurora：`/scratch/pawsey0115/hwang4/miniconda3/envs/aurora-gpu`（PyTorch；仅 GPU 推理用，登录节点阶段复用 infer-gpu）

---

## 7. 常见问题

| 现象 | 原因 / 处理 |
|------|-------------|
| `import jax` 卡死 | jaxlib `.so` 损坏 → `pip --force-reinstall --no-deps jaxlib==0.4.35`；或冷 GPU，先 `rocminfo` 预热（sbatch 已内置） |
| Slurm 报 `memory ... rejected` | GPU 分区按卡分配内存，不能用 `--mem`；需增加 `--gpus-per-node` |
| 0.25° 推理 OOM | 至少 3 GCD（86 GiB）；或 `--exclusive` 独占整节点 |
| 下载 `503 Slow Down` | AWS 源限流，默认已用 `google`；可换 `--source` |
| 推理后找不到预测 | 预测已改到外部 `results/` 树（非旧的 `predictions/`），见 §4 |
| `./realtime_all.sh` 提示 `Permission denied` | 可执行位丢失，`chmod +x realtime_all.sh` 后用 `./` 或直接 `bash realtime_all.sh` |
| 两个模型 cycle 不一致 | 必须从顶层 `realtime_all.sh` 提交；该入口只解析一次 `--latest`。单独执行两个模型时应显式传同一组 `--date/--time` |

---

## 8. 运行验收

一次双模型运行只有同时满足以下条件才算成功：

1. 顶层日志以 `SUCCESS` 结束，GraphCast 与 Aurora 的 Slurm 作业均为 `COMPLETED`；
2. 同一 `<init>Z` 下两个模型各有一个非零、可被 xarray 打开的中国区 `.nc`；
3. 两个文件的 `convention=unified-forecast-1`、`init_time`、40 个步长和 240 h 预报时效一致；
4. 两套 `series/`、`overview/`、`gif/` 均生成，且不存在残留的 `*.tmp.nc`。

NetCDF 先写入同目录临时文件，成功关闭后再原子替换正式文件；节点故障或磁盘写满不会留下一个看似正式、实际损坏的预测文件。
