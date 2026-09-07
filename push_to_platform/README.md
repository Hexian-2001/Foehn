# push_to_platform

把 GraphCast-operational / Aurora-0.25-finetuned 的预报结果推送到气象大模型评测平台（`benchmark.mingyangai.com.cn`）。

## 适配决策（开源模型 → 平台 1h/72h 协议）

| 项 | 开源模型 | 平台要求 | 本实现 |
|---|---|---|---|
| 时间分辨率 | 6h（40 steps × 6h = 240h） | 1h × 72（未来 3 天） | 6h→1h 线性插值，取前 72h |
| 起报对齐 | 任意 cycle | 固定 00Z | 只推 `{date}T00Z` cycle |
| lead 1–5 | 模型无 lead-0 输出 | 需要 1..72 | 用 IFS fc0 分析场（stage-2 输入 `.nc` 的 lead-0）做锚点，线性插值 0→6h；缺失时退化为零阶保持 |
| 100m 风速 | 无直接 100m 变量 | `wind_speed_100m_ms` | 1000 hPa 层风速 `√(u²+v²)`（项目既有约定） |
| 10m 风速 | `10m_u/v_component_of_wind` | `wind_speed_10m_ms` | `√(10m_u² + 10m_v²)` |
| 风向 | u/v 分量 | `wind_direction_deg`（可选） | 气象风向 `(180° + atan2(u,v)) mod 360` |
| 场站级坐标 | — | farm（整场） | 取风场 centroid 最近网格点 |
| 风机级坐标 | — | turbines（逐台） | 取每台风机 lat/lon 最近网格点（**不做降尺度**，同场多台风机常落到同一网格） |
| 光伏 | 无 GHI/短波辐射 | `ghi_wm2`（SP-01/SP-02） | 全部置 null（暂不推光伏，协议仍要求站点存在） |

## 目录结构

```
push_to_platform/
├── asset_catalog.json   # 平台资产清单快照（catalog_version 1.0.0，8 风场 180 风机 + 2 光伏）
├── config.py            # 路径 / 平台 URL / 模型注册表 / API key
├── assets_map.py        # 读清单 → 站点 & 风机坐标
├── extract.py           # 读 .nc → 取点 → 6h→1h 插值 → 100m/10m/风向
├── payload.py           # 组装 ForecastPushRequest JSON
├── push.py              # HTTP POST + 幂等 + 回执
├── cli.py               # 命令行入口
└── realtime_push.sh     # 每日 00Z 推送（Pawsey）
```

## 本地试跑（dry-run，不推送）

```bash
cd forecast_models
PYTHON=/d/conda_envs/mymet/python.exe   # 或任意含 xarray+numpy 的环境

# 生成 payload 到文件（不联网）
$PYTHON -m push_to_platform.cli --model graphcast --date 2026-08-27 --dry-run --out graphcast_payload.json

# 直接打印 payload
$PYTHON -m push_to_platform.cli --model graphcast --date 2026-08-27 --dry-run
```

## 真实推送

```bash
export BENCHMARK_PUSH_URL="https://benchmark.mingyangai.com.cn"
export BENCHMARK_PUSH_KEYS="graphcast=sk-graphcast-xxx,aurora=sk-aurora-xxx"

$PYTHON -m push_to_platform.cli --model graphcast --date 2026-08-27
$PYTHON -m push_to_platform.cli --model aurora   --date 2026-08-27
```

`--submission-type backfill` 用于历史回灌（不触发“迟到”告警）。

## Pawsey 每日推送

在 `realtime_all.sh` 的推理阶段之后追加一行（或单独 cron）：

```bash
bash push_to_platform/realtime_push.sh   # 默认推今天 00Z；可带日期参数
```

`realtime_push.sh` 对每个模型独立 try，单个失败不中断另一个；退出码非 0 即“未推送”。

## 前置条件（平台侧，需先配好）

1. 平台 `.env` 注册两个供应商，与 `push_to_platform/config.py` 的 `provider` 名一致：

   ```
   BENCHMARK_PROVIDER_KEYS=...,graphcast=sk-graphcast-xxx,aurora=sk-aurora-xxx
   ```

2. 本仓库的 `asset_catalog.json` 与平台 `catalog_version`（1.0.0）保持一致；
   平台清单若升级，需重新拷贝 `benchmark_platform/assets/asset_catalog.json` 覆盖本快照。

## 说明 / 限制

- **同场多台风机同一网格值**：0.25°≈27km，风机间隔约 250m，未做降尺度时同一风场多台风机取值相同，属预期。
- **lead 1–5 依赖 stage-2 输入 `.nc`**：`weathernext_forecast/data/processed/source-ifs_date-*.nc`（GraphCast 预处理产出，含 lead-0 分析场）。缺失时自动退化为零阶保持（lead 1–5 = lead 6），不影响推送但精度略降。
- **分析场为 IFS fc0**，与模型预报在 lead 6 处会有小幅跳跃（模型自身偏差），属正常。
- 幂等：同一天同一 `request_id` 重复提交返回 `duplicate`；修正数据请用 `--request-id-suffix` 换新 id。
