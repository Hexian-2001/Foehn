# Aurora 0.25 finetuned — 实时推理操作手册

Microsoft Aurora（PyTorch，非 JAX）经典 0.25° IFS HRES T0 微调版，与 GraphCast
解耦、共享统一的预测存储与可视化管线。

## 架构

```
opendata_download          (共享，模型无关)  下载 ECMWF open-data GRIB
        │
        ├─ data_processing (GraphCast)        GRIB → GraphCast 输入 .nc
        └─ aurora_forecast.adapter  (Aurora)  GRIB → Aurora 输入 .nc
                  │
                  ▼
   aurora_forecast.inference (GPU, torch)      rollout → 统一预测 .nc
                  │
                  ▼
   weathernext_forecast.scripts.visualize (共享)  统一 .nc → 图表/GIF
```

- **变量契约**（probe 已锁）：面 `t2m/u10/v10/msl → 2t/10u/10v/msl`；大气
  `z/t/u/v/q`（丢弃 `w`）；lat 递减（不翻转）、lon roll 后 [0,360)、level 递增。
- **静态场**来自 pickle `{z, slt, lsm}` (721,1440)，非 ERA5。
- **输入 .nc** 采用 Aurora 自己的 `Batch.to_netcdf` 布局，GPU 端用
  `Batch.from_netcdf` 原样读回。
- **输出**统一名（`2m_temperature`/`10m_u_component_of_wind`/...），落
  `results/aurora/0.25-finetuned/<init>Z/predictions/*.nc`。

## 一次性部署（Setonix）

```bash
# 1. 推送源码（scripts/src/upstream 包；权重不推，>1GB）
./sync.sh push

# 2. 建 torch 推理环境（仅 GPU 推理用；下载/适配/可视化复用 infer-gpu）
bash aurora_forecast/scripts/setup_env.sh

# 3. 下载权重（finetuned.ckpt ~4.7GB + static.pickle ~12MB）
bash aurora_forecast/scripts/download_weights.sh
```

## 实时推理

```bash
# 单模型：Aurora 端到端（下载→适配→Slurm→可视化）
./aurora_forecast/scripts/realtime.sh                    # 最新 cycle
./aurora_forecast/scripts/realtime.sh --date 2026-08-31 --time 00

# 双模型：GraphCast + Aurora 一个命令跑完
./realtime_all.sh                                        # 最新 cycle
./realtime_all.sh --date 2026-08-31 --time 00
```

资源：Aurora 只需 **1 GPU**（MI250X 双 GCD，gfx90a 原生支持，无 HSA override）。
GraphCast 需 3 GPU。两者用不同 conda 环境（infer-gpu / aurora-gpu），sbatch 自动切换。

## 关键 env 变量

| 变量 | 作用 | 默认 |
|---|---|---|
| `AURORA_INPUT_FILENAME` | sbatch 指向的输入 .nc（realtime.py 注入） | — |
| `AURORA_PREDICT_REGION` | 落盘裁剪区 `china`/`global` | china |
| `AURORA_FORECAST_STEPS` | 预报步数（6h 步长） | 40（240h） |
| `RESULTS_ROOT` | 统一结果树根 | `<repo>/results` |
