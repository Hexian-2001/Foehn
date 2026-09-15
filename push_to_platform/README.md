# push_to_platform

把 GraphCast-operational / Aurora-0.25-finetuned 的预报结果推送到气象大模型评测平台（`benchmark.mingyangai.com.cn`）。

## 适配决策（开源模型 → 平台 1h/72h 协议）

| 项 | 开源模型 | 平台要求 | 本实现 |
|---|---|---|---|
| 时间分辨率 | 6h（40 steps × 6h = 240h） | 1h × 72 | 6h→1h 线性插值 |
| 起报时刻 | 任意 cycle | 固定 `16:00:00Z`（次日北京时间 00:00） | `start_date` 恒为 `T16:00:00Z`；按 `init_time` 计算 lead 偏移 |
| lead 对齐 | 模型 lead 6..240 | 下标 0 = start_date 起报，1..72 | `lead_start = (start_date − init_time)`，把模型 6h 步长插值到 `[lead_start .. lead_start+71]` |
| 起报场选择 | 多个可用 cycle | — | 取 `init_time <= start_date` 的最新预测（`find_prediction_nc`） |
| 100m 风速 | 无直接 100m 变量 | `wind_speed_100m_ms` | 1000 hPa 层风速 `√(u²+v²)`（项目既有约定） |
| 10m 风速 | `10m_u/v_component_of_wind` | `wind_speed_10m_ms` | `√(10m_u² + 10m_v²)` |
| 风向 | u/v 分量 | `wind_direction_deg`（可选） | 气象风向 `(180° + atan2(u,v)) mod 360` |
| 场站级坐标 | — | farm（整场） | 取风场 centroid 最近网格点 |
| 风机级坐标 | — | turbines（逐台） | 取每台风机 lat/lon 最近网格点（**不做降尺度**） |
| 光伏 | 无 GHI/短波辐射 | `ghi_wm2`（SP-01/SP-02） | 全部置 null（暂不推光伏，协议仍要求站点存在） |

## 目录结构

```
push_to_platform/
├── asset_catalog.json   # 平台资产清单快照（catalog_version 1.0.0，8 风场 180 风机 + 2 光伏）
├── config.py            # 路径 / 平台 URL / 模型注册表 / FORECAST_ISSUANCE_UTC / API key
├── assets_map.py        # 读清单 → 站点 & 风机坐标
├── extract.py           # 读 .nc → 按 lead_time 取点 → 6h→1h 插值 → 100m/10m/风向
├── payload.py           # 组装 ForecastPushRequest JSON
├── push.py              # HTTP POST + 幂等 + 回执
├── cli.py               # 命令行入口
└── realtime_push.sh     # 每日推送（Pawsey，由 scripts/daily_run.sh 调用）
```

## 本地试跑（dry-run，不推送）

```bash
cd forecast_models
PYTHON=/d/conda_envs/mymet/python.exe   # 或任意含 xarray+numpy 的环境

# 不指定 --start-date：默认明天 16:00Z（当天 UTC 的 start_date）
$PYTHON -m push_to_platform.cli --model graphcast --dry-run

# 指定起报时刻（用于回灌 / 对历史预测验证）
$PYTHON -m push_to_platform.cli --model graphcast --start-date 2026-09-04T16:00:00Z --dry-run --out graphcast_payload.json
```

## 真实推送

```bash
export BENCHMARK_PUSH_URL="https://benchmark.mingyangai.com.cn"
export BENCHMARK_PUSH_KEYS="graphcast=sk-graphcast-xxx,aurora=sk-aurora-xxx"

$PYTHON -m push_to_platform.cli --model graphcast   # 默认明天 16:00Z
$PYTHON -m push_to_platform.cli --model aurora
```

`--submission-type backfill` 用于历史回灌（不触发“迟到”告警）。

## Pawsey 每日自动运行

`scripts/daily_run.sh` + `scripts/daily_run.sbatch` 组成自续排程（Setonix 无 cron）：

1. **先 resubmit** 次日 02:00 AWST 的作业（`--begin`），保证排程在任何失败下都延续；
2. 跑 `realtime_all.sh --latest`（下载 + 处理 + GPU 推理，GraphCast + Aurora）；
3. 跑 `push_to_platform/realtime_push.sh`（抽取 + 推送，两模型独立 try）。

时间语义（集群 TZ = AWST = UTC+8）：

- `02:00 AWST = 18:00 UTC`——刚过 12Z cycle 的 ~6h 可用延迟，`--latest` 解析到 12Z，推送用最新初始场；
- 推送截止 `08:00 AWST = 00:00 UTC`，起报 `16:00 UTC`（次日北京时间 00:00）；`realtime_push.sh` 不带参数时自动取「明天 16:00Z」。

首次提交（在 login 节点）：

```bash
cd /scratch/pawsey0115/hwang4/projects/Foehn
sbatch --begin=2026-09-15T02:00:00 scripts/daily_run.sbatch   # 或直接 sbatch 立即首跑
```

## 前置条件（平台侧，需先配好）

1. 平台 `.env` 注册两个供应商，与 `push_to_platform/config.py` 的 `provider` 名一致：

   ```
   BENCHMARK_PROVIDER_KEYS=...,graphcast=sk-graphcast-xxx,aurora=sk-aurora-xxx
   ```

2. 本仓库的 `asset_catalog.json` 与平台 `catalog_version`（1.0.0）保持一致；
   平台清单若升级，需重新拷贝 `benchmark_platform/assets/asset_catalog.json` 覆盖本快照。

3. `push_to_platform/.env`（Pawsey 本地，已 gitignore）：

   ```
   BENCHMARK_PUSH_URL=https://benchmark.mingyangai.com.cn
   BENCHMARK_PUSH_KEYS=graphcast=sk-...,aurora=sk-...
   ```

## 说明 / 限制

- **同场多台风机同一网格值**：0.25°≈27km，风机间隔约 250m，未做降尺度时同一风场多台风机取值相同，属预期。
- **起报场必须 ≥ 6h 先于 start_date**：模型首个 step 是 lead 6，`lead_start = start_date − init_time < 6` 时会报错（每日流程不会发生：cycle 可用延迟 ~6h、起报在次日 16:00Z）。
- **窗口越界即失败而非静默外推**：`lead_start+71 > 240`（起报场太旧）会直接报错，提示改推更新的 cycle。
- 幂等：同一天同一 `request_id` 重复提交返回 `duplicate`；修正数据请用 `--request-id-suffix` 换新 id。
