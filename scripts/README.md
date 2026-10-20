# CLI 入口索引

`scripts/` 中的文件是仓库根目录下运行的薄入口，不是可 import 的业务模块；可复用逻辑放在
`agent_company/`。不要根据脚本名字推断是否付费，以下表格为准。结果默认写入被 Git 忽略的 `runs/`。

## 正式运行（付费，统一经 formal 启动器）

| 入口 | 用途 |
| --- | --- |
| `launch_programbench_formal.py` | 正式系列唯一入口，每次只做一个动作：`--deploy --commit <完整SHA>`、`--verify`、`--prepare`、`--launch`、`--smoke`、`--probe`、`--status` |
| `run_programbench_policy_comparison.py` | Single/Team 推理加官方评分；由启动器调用。默认只读计划，`--run` 才运行 |
| `run_programbench_evolved_followup.py` | 演化 followup 轮；由启动器调用 |
| `s0_native_smoke.py` | 选定一臂的原生机制冒烟：`--scripted` 用脚本 provider（零 API），`--real` 调用配置模型；只证接线，不产出分数 |
| `s0_cache_probe.py` | 最多 6 次的缓存诊断，任一请求失败立即停；只存元数据，不存生成文本 |
| `launch_programbench_s0_deepseek.py` | 旧 S0 启动器，已不用于正式运行；formal 启动器仍读取其中的策略常量 |
| `programbench_docker_sudo.sh` | 服务器上可选的项目内 docker 包装：保持当前用户、只为该子进程临时加 docker 组 |

启动器规则：

- 部署按提交原样归档，不加工作区覆盖，不 push。本系列第一轮、全部确认对和 followup 必须共用同一次
  部署，中途重部署按新条件处理。
- `--prepare/--launch` 另传 `--runner comparison|followup --step <步骤> --profile <路径>` 和显式
  `--repeat-index`：S0 用 0，第一轮 off/global/master/propose 分别为 1/2/3/4；确认、批准派发和
  followup 用本部署 `paid_runs` 里未用过的正整数。
- `prepare` 不读 stdin、不写文件；`launch` 检查通过后才读 stdin 凭据，并独占创建步骤目录。
- 零 API 冒烟：`--smoke --scripted --arm <臂> --sequence <正整数>`；真实冒烟改为 `--real`，固定
  `--max-calls 40 --timeout 420`。相同目录已存在就拒绝，不重试。
- `--status` 只读进程、progress 和官方结果字段，不自动恢复运行。

## 判读与读数（不联系 provider）

| 入口 | 用途 |
| --- | --- |
| `read_programbench_total_comparison.py` | `--incumbent <trial> --candidate <trial>` 产生触发报告；确认时另传 `--trigger <saved-report.json>`，复用保存的门槛 |
| `check_native_final_reports.py` | 检测疑似最终汇报丢失，缺证据保持 unknown，不改官方分数 |
| `report_programbench_s0.py` | 从保存的 events.jsonl 离线读 S0 指标 |
| `project_programbench_ac_evidence.py` | 把已有 Team 证据投影成不含推理内容的视图 |
| `summarize_programbench_baseline.py` | 汇总本地 comparison 结果目录，输出描述性 JSON |
| `summarize_programbench_score_layers.py` | 读过滤后的评估 sidecar：`--names-only` 只出测试名，`--layer-rules` 出分层 |

followup 的协议参数：显式 `--repeat-index`、`--evolution-round 1|2|3`；后续轮用 `--previous-round`
提供完整前序轮次（completed 或 blocked_*，且 running=[]）。`--approved-trial ROUND=PATH` 绑定经用户
批准派发的已有 trial；`--history-trial` 供同条件 Team 极差判读。以上工具都不自动启动、采纳或部署。

## 准备（不联系 provider）

| 入口 | 用途 |
| --- | --- |
| `prepare_programbench_first_round.py` | 打印第一轮 off→global→master 与 Single+propose 的波次；默认只读 |
| `prepare_programbench_baseline.py` | 已推后的 K=3 Pandoc 基线各轮 profile；`--write-profiles` 才写到新目录 |
| `prepare_stage1_arm64_node.py` | 为 ARM64 本机准备 Node 运行时，不改 amd64 任务镜像 |

## 早期 RoadmapBench 入口

| 入口 | 用途 |
| --- | --- |
| `run_stage1.py` | download/inspect 不调用模型；run-task、baseline、run（联合演化）调用所配置 provider |

V4/V5、早期 protocol/evolution、qualitative triad 与一次性的 `check_programbench_*` 验证脚本已于
2026-10-15 删除，需要时从提交 `e6a9790` 恢复。
