# Stage-1 harness 迭代方案：角色级通信策略与独立运行时改进（2026-10-10，修订 v3.3，待审核）

状态更新（2026-10-14）：v3.4 已随 `2f7bd45` 提交并 push，10-13 Kimi/FFmpeg 部署完成两模式零 API 冒烟；取消人数上限 `2a3382e` 及新的 DS 流式准备尚无服务器验证。本文保留设计与历史审查，具体推进与授权范围见 `stage1_next_steps_plan_20261014.md`。

v3.3 保留 v3.2 的核心规则：**relay 是策略状态的唯一权威，原生消息仅负责通知与尝试唤醒**。本次补充确定性注入与缓存计量、全局广播通知、`handle()` 返回前的晚提交、统一可选 base、逻辑请求关联字段、整响应拒绝的连带成本，以及四种模式的披露草稿（§9）。提案、角色策略、全局策略和决定以整批持久提交生效，不因后续消息失败或缺确认回退。修订原因和历史对应见 §8。

研究问题来自与导师的讨论：(1) master 为不同 subagent 设计不同通信策略；(2) 比较 high-level 与 individual-level；(3) 比较 master 集中修改与 worker 提议、master 审批。当前 master 已能在 spawn prompt 里给不同角色不同规则，worker 也能用消息提建议；缺的是**显式策略对象、受控修改入口、持久版本和策略—行为关联**。

v3 相对 v2 的核心变化：**不引入 MCP，不加新工具，不做角色级调用认证**。策略的设定、提议、决定全部复用原生 `send_message`，由 relay 在带有运行凭据与身份一致性检查的模型请求里识别固定前缀。v2 的状态语义（槽/实例分离、继任、版本与 stale 检查、全样本保留、注入与唤醒分离、证据分阶段）全部保留。v2→v3 的逐项对应见 §8。v3.3 的原生接入还包括广播收件人/发送结果的元数据埋点，另保留 D10 的可选判断；不改原生广播路由、不另造唤醒通道。

---

## 0. 结论与需要拍板的点

**最小实现**：relay 识别四种带前缀的 `send_message`（设定角色策略、提议、决定、可选的全局策略在线修改）和 spawn prompt 里的两个标记（初始角色策略、继任）。`prepare_response()` 仅校验并暂存策略变更；本地响应/尝试/usage 证据成功写入后，在 `handle()` 返回前同锁复核并一次性写入策略事件批次，发布状态。全局更新必须广播，角色更新仍定向；逐收件人通知结果不决定状态是否有效。每次 coding 请求注入当前策略，master 还获得待决提案的编号与完整正文，worker 获得自己提案的状态。注入块是状态的确定性表示；缓存命中及缺失单列。提案消息不改写；投影区分提交、payload 注入、上游尝试、原生通知和按请求版本关联的行为。

**不列为前置**：provider 恢复、轮询治理、compaction。独立验收；纳入正式实验时三臂统一。

**2026-10-10 用户决定**：D1、D4、D9 按建议；D10 选 C（compat 一行判断，§9.3 采用 D10-C 尾注）。S1 已实施，见 §8 末尾的 S1 实施记录。

需要你决定的点（§7 展开）：

| # | 决定 | 我的建议 |
|---|---|---|
| D1 | 角色策略与 C 的优先级 | 对该角色的可变通信条款，已批准角色策略优先；工具权限、安全边界、取消、硬截止不可覆盖 |
| D2 | 入口 | **已定：`send_message` 前缀约定**。S0 若模型遵从率不可接受，升级为 compat 注册原生工具；MCP 不作首选 |
| D3 | 首轮臂 | H-adaptive / R-master / R-propose（见 D9）；只主张"机制包"与"提案入口"效果 |
| D4 | 继任 | spawn prompt 里 `SUCCEEDS: <name>`；只允许继任已终止实例；槽与版本延续、pending 作废 |
| D5 | 原生提示与策略关系 | 注入文本声明通信条款优先级；不声称能关掉原生自动汇报；横向消息多少不是目标 |
| D6 | 运行时范围 | 先做角色最小机制；R1/R2/R4 分开验收 |
| D7 | 顺序 | S0 约定遵从性冒烟 → S1 状态语义 → S2 relay 识别/注入/spawn 标记 → S3 投影 → S4 独立 runtime → S5 实验准备 |
| D9 | 基线臂 | **H-adaptive**（master 可用 `GLOBAL-POLICY` 在线改 C）替代静态 H 作正式基线；要成立须把 master 的 C 从静态提示移到 relay 注入（§2），spawn 初版的不对称保留并计数 |
| D10 | 提案消息是否压掉原生最终汇报（已定 C；实测见 §10.1，已测场景中未发生） | 建议在 compat 加一行判断：首行为 `POLICY-PROPOSAL` 的消息不计入原生"本回合已显式汇报"集合。代价是放弃"零原生接入"的说法；替代是只披露，但那会给 R-propose 臂的 worker 加一条行为要求，构成臂间混杂 |

v2 的 D8（MAC 还是 nonce）随 MCP 一起取消。

---

## 1. 现状与依据

### 1.1 策略层次

| 层 | 内容 | 谁看到 | 注入位置 | 代码 |
|---|---|---|---|---|
| A | 分工/角色/所有权 | master | master `--system-prompt`（静态） | `qwen.py` `master_system()` |
| C | 触发/收件人/内容 | master + 每个 worker | master 静态；worker 每次 coding 请求由 relay 追加 | `relay.py` `payload()` 约第 377–380 行 |
| 临时角色指令 | "只向我汇报""先问 probe" | 单个 worker | spawn prompt 或后续消息；无版本对象 | `agent.prompt` / `send_message` |

- 抽取报告（`research_notes/team_prompt_stack_原文与抽取_20261005.md`）记录 decoder3 的"只向我汇报"与 C 的"直接路由"冲突，没有生效版本和覆盖语义可查。
- 质量分析 B4：decoder→decoder2→decoder3 各拿新 prompt；没有自动继承机制，master 只能手工重述。
- B5：12 次运行无 compaction。注入解决的是持久呈现，不是已经发生的丢失。
- `programbench_evidence_projection.py` 已有 `by_role`、`collaboration_timeline`、`delivery`（按 `message_id` 关联 queued/observed，第 293–330 行）、`checkins`、`runtime_rules`；缺的是策略版本与请求/行为的关联。B3 的 observed/queued（72/82、110/142、132/143）可以继续用，但要把"晚到"与"从未 observed"拆开报告。

### 1.2 Qwen 原生 teammate 提示的偏置

原生提示要求先查任务板、向 leader 汇报，回合末文本自动转给 leader（执行机制，文本不能关）。横向消息占比 P0 28%、P1 10-01 24%、10-02 约 5%、rerun2 3 条，作为动机保留；它和 A/C、分工、存活、provider 状态混在一起，不能当原生提示压制角色策略的证据。横向比例不是优化目标。

### 1.3 relay 已有的、v3 直接依赖的机制

- **请求身份**：`caller()` 用 compat 层的 owner 头和系统提示里的身份声明互相核对，得到每次 coding 请求的 `role`。这是**共享运行凭据下的一致性检查**，不是角色级强认证；整个实验的工具过滤和预算都建立在这一层上。v3 的策略入口发生在这些请求的响应里，不新增身份来源，但新增的策略写入是新的权限敏感行为，与现有工具过滤同一信任层（§3.5）。
- **响应阶段校验**：`prepare_response()` 在执行前看到每条工具调用的参数，能改写（spawn 的 `working_dir`）或整体拒绝（`rejected_tool_response()`，模型看到原因后重发）。v3 在这里识别前缀。校验发生在执行前，**不等于执行成功**：之后还有响应写回、原生守卫、取消和实际发送。
- **执行结果可能迟到或缺失**：发送者下一次请求通常带这条调用的工具结果，但要等该轮工具批次结束；回合终止等情况也可能没有下一次请求。原生单播成功文本是 `Message sent to "<to>".`；失败是 `Failed to send message: …` 或 `Error: …`。通知关联将使用新增 `source_request` 字段与已有 `tool_call_id`，不是复用一个已经存在的逻辑请求字段（见 §3.1）。成功只说明原生发送成功，不证明接收者已发起模型请求；缺结果记 unknown，不撤销策略状态。
- **原生广播的实际范围**：`send_message(to="*")` 遍历发送时成员表、排除发送者，不过滤 terminal；master 广播并不等于“仅向存活 worker 发送”。原生返回 `total` 与 `failedRecipients`，没有成功名单；部分失败仍可返回普通工具结果，零收件人也可能显示广播成功。因此需要在 compat 的实际枚举/结算点记录收件人及逐项结果，仅凭总体成功文本不能宣称所有 worker 已送达（§3.2）。
- **广播关联尚非现成功能**：scheduler 有 `request.callId`，但当前 `SendMessageInvocation.execute(signal)` 和 `TeamManager.broadcast(message, fromName)` 未接收它，现有进程内上下文也未传该字段。广播埋点可先提供收件人级发送事实；与某次策略更新可唯一关联时才归因，否则为 unknown。若实施时需要透传已有 callId，只做进程内关联并用真实 CLI 脚本验证，不假设当前 hook 已具备，更不新增消息改写、认证信封或唤醒通道。
- **校验之后仍可能本地失败**：`_attempt()` 中 `prepare_response()` 之后还有脱敏、model views、响应文件、attempt 和 usage 写入；`handle()` 的 finally 又写一次 usage，失败会将成功改成 424。晚提交必须越过这些写入，不能只移到 `_attempt()` 尾部；HTTP 写回仍在 `handle()` 返回之后。
- **原生消息文本约定已有先例**：Qwen 自己用消息正文识别 `shutdown_approved`（`sendMessage()` 里的 `classifyShutdownResponse`），我们的 master 提示也要求 worker 这样确认关闭。前缀约定不是新的交互范式。
- **原生投递有两条路**：发给 worker 的消息进 `pendingMessages` 队列，compat 在这里有 `team_message_queued/observed` 埋点；发给 leader 的消息写磁盘 inbox、由轮询读入再经 `leaderMessageCallback` 进入 master 请求，**没有埋点**。master 看到的形式是 `<teammate_message from="<name>">…</teammate_message>` 信封。
- **原生副作用（10-11 按 §10.1 更正）**：任何非自动的成员→leader 消息都会执行 `explicitLeaderReports.add`；但原生每次响应若有非空正文，就会把它设为最终汇报并清掉这个标记。已测场景中，发过 leader 消息之后的非空正文照常被转发；标记只在之后没有非空正文时才可能起作用，这种情况是否出现、出现时的影响尚未测到。
- **原生同名 spawn 被拒绝**：`generateUniqueTeammateName` 对已存在的名字抛错"A teammate named X already exists"，终止成员仍留在名册。relay 的 `register()` 对同名幂等且先发 `worker_registered`，所以现状下同名重生会留下一条原生从未创建的注册记录。
- **生命周期事件的去向**：compat 的 `record()` 只写 stderr；`qwen.py` 的 `native_lifecycle()` 在 CLI 退出后才解析。stderr 由 `workspace.py` 的 `Popen` 直接写到**宿主机文件**，relay 在同一台机器上。
- **历史还原**：现有 spawn 参数改写等仍使用 `restore_model_history()`。本方案不向提案消息添加编号，不新增提案历史还原依赖；编号由双方的 system 注入提供。

---

## 2. 研究轴与臂

导师的两条轴：high-level vs individual-level；centralized evolve vs subagent proposes。已确认：worker 只提自己的；必须 master 批准；不自动批准。

| 臂 | 策略对象 | master 可用前缀 | worker 可用前缀 | 对应轴 |
|---|---|---|---|---|
| H（静态，可选） | 无 | 无 | 无 | 现状 |
| **H-adaptive（建议基线）** | 全局 C 的在线版本 | `GLOBAL-POLICY` | 无 | 高层策略，master 可在线改 |
| R-master | 每角色策略 | `ROLE-POLICY`（含 spawn 标记） | 无 | 角色级，master 集中改 |
| R-propose | 每角色策略 | 上述 + `POLICY-DECISION` | `POLICY-PROPOSAL` | 角色级 + worker 提议、master 批 |

- **H-adaptive→R-master** 比较的是粒度（全局 vs 角色）；**R-master→R-propose** 比较的是正式提案入口的增量。静态 H 只用于回答"在线修改本身有没有用"，首轮可不跑。
- H-adaptive 要成为对等的粒度对照，需要三件事，不是只加一个前缀：(a) master 的 C 现在写死在 `--system-prompt` 里，更新后它会同时看到旧版与新版；所以在 `global/master/propose` 三种模式下都把 C 从 master 静态提示移到 relay 注入（与 worker 同一机制，带版本号），静态提示只留"由运行时提供"的说明。(b) 全局更新与角色策略一样，以 relay 持久提交为生效点，后续请求注入新版；收件人的原生消息仅作通知，通知成功不作为任何角色更新的前提。(c) R-master 的 spawn 初版是 H-adaptive 没有的修改机会，这个不对称保留，按 set 事件的时间与 token 计数报告，不宣称已消除。
- **通知覆盖对齐**：H-adaptive 的 `GLOBAL-POLICY` 必须由 master 发给 `"*"`，尝试通知原生成员表中所有受影响 worker；R-master 的 ROLE-POLICY 定向通知受影响槽的当前实例。广播不保证全部成功或同时唤醒，terminal 成员、零收件人、部分失败和缺证据分别计量。提交后才加入的 worker 从首次请求注入获得当前 C，不补广播、不额外启动模型。这里只对齐通知意图的覆盖范围，不声称实际唤醒机会已完全相等。
- 三臂同一份 A/C 起点、同一 runtime、同样的任务与预算；前缀约定的说明文本是机制的一部分，其 token 一并计量。
- **缓存也是机制成本**：动态状态位于第一条 system 消息，状态变化可能缩短可复用前缀，master 尤其受任一槽/提案变化影响。各臂同时报告 prompt/cached/uncached tokens、缓存指标覆盖率和延迟；总成本差异包含通信行为、注入长度与缓存变化，不把它全归因于通信效率。静态 H、H-adaptive 与角色臂的变化频率不同，即使固定模型/墙钟也不能忽略此混杂（§3.3.1）。
- 所有臂的 worker 仍可用普通消息提建议；R-propose 测的是"可追踪、需审批"的通道增量。普通建议与正式提案在投影里分开标注。
- **策略是文本规范，不是工具能力开关**：relay 控制谁能写入正式策略状态，但不强制模型遵循其中的自然语言通信规则。worker 本来就能按既有工具权限发消息；批准代表策略获得正式认可，不代表此时才获得发送能力。R-propose 研究的是正式批准的自我策略是否改变行为，不能仅凭批准率声称实现了运行时通信约束。
- **只提自己的发送规则**：正式提案范围是本角色何时发送、发给谁、发送什么；不授权改别的角色的规则。希望其他角色改变做法时，worker 仍用普通消息请 master 处理，不计为自己的正式提案。自然语言是否越过该研究范围作错误分析，不声称身份检查能判定所有语义越权。披露要求待决期间按当前生效策略继续工作，不必停工等待审批。
- **所有预分配的运行都进主表**（含未使用约定、未批准、失败）；未启动/无评分单列，不补零。使用率、提案决定率、批准率、版本注入覆盖率作独立诊断。
- 主结果：功能层分数；失配指标与策略指标为次要/机制指标。任务选择、重复次数、统计方法在付费实验前另定。

---

## 3. 设计

### 3.1 状态对象

```text
RoleSlot
  slot_id                 run 内稳定；首任 worker 名
  current_instance        当前绑定的 worker 名；空 = 暂无
  current_version         0 = 只有 C；每次批准修改递增

RoleBinding
  instance                run 内唯一的 worker 名；当前原生实现不允许同名重生
  slot_id
  lifecycle               planned | active | terminal
  predecessor             可空；由 SUCCEEDS 产生

RolePolicyVersion
  slot_id, version, text（≤2000 字符）
  provenance              master_set | proposal_approved | proposal_amended | spawn_initial
  proposal_id             可空
  actor, reason, committed_at, source_request（新增关联字段，取准入时的逻辑请求号）

PolicyProposal
  proposal_id, slot_id, proposer（instance）
  base_version            提议消息所属请求实际注入的版本
  change, reason, evidence 完整提案正文，三段及标签合计 ≤2000 字符
  created_at, source_request, tool_call_id
  status                  pending | approved | rejected | amended | voided
  decided_at, note, decided_version

GlobalPolicyVersion（仅 H-adaptive）
  version, text, actor, reason, committed_at, source_request
```

**逻辑请求关联（新增字段，不新增一套身份系统）**：当前 relay 只有 `_logical_model_calls` 计数和各 attempt 的 `attempt/retry_index/upstream_started`，没有 `source_request` 字段。在现有准入锁内递增逻辑计数时捕获该值，作为 run 内逻辑请求号写入新增 `source_request` 字段，关联固定 payload 快照、各次 retry、策略批次与通知的 `tool_call_id`；不新增 UUID、哈希或 `operation_id`。即使 `max_model_calls=null` 也记录关联，不能依赖只在设预算时才出现的 admitted 事件。未准入的 payload 只记 prepared/unadmitted，不伪造逻辑请求号或上游启动。

规则：

- 每槽同时最多一个 active 实例、一个 pending 提案。
- **名字在整个 run 内唯一**：spawn 名字与任一 planned/active/terminal 实例重名即在 relay 拒绝（原生也会拒绝，但 relay 先拒可避免留下无效注册）。继任者必须用新名字。
- `planned` 在含 spawn 的响应完成晚提交时产生，而不是在初次校验时产生；该名字的**第一次 coding 请求**到达 relay 时变 `active`（这就是绑定点，不需要原生实例 ID）。master 后续请求里若该 spawn 的工具结果是错误，planned 作废；运行结束仍无首次请求也作废。
- `terminal` 的**实时来源**：relay 增量读取宿主机上的 `qwen_stderr.txt`，消费 `Agent Company lifecycle:` 行里的 `worker_terminal`（以及 `team_message_queued/observed`）。按行号/时间去重，延迟与丢行都要测；日志里有事件不等于 relay 状态已更新，所以 `SUCCEEDS` 校验以 relay 已消费到的 terminal 为准，未消费到则拒绝并说明"前任终止尚未被运行时确认，稍后重试"。实例 terminal 时其 pending 置 `voided(proposer_terminal)`。
- 继任：`SUCCEEDS: <name>` 指向已 terminal 的前任；绑定同一槽、沿用当前版本、不新建版本、不继承 pending。继任不恢复会话、不复制工作区、不改任务归属。`SUCCEEDS` 与 `ROLE-POLICY:` 块同时出现则拒绝。
- **由模型响应产生的策略写入及 spawn/继任 planned 变更**，在 `handle()` 的晚提交锁内重查绑定、生命周期、pending 与请求快照的 base，并检查现有取消/硬截止条件；初次校验之后其他角色可能已修改状态，不能直接提交旧结果。按最新状态重新模拟整批，通过才将关联变更作为一条事件批次持久化，再发布缓存；失败不暴露新状态，重放不恢复半个响应（如 set v2 与作废旧 pending 不得拆开）。**首次请求的 planned→active 绑定与 stderr 生命周期输入**按各自输入时点，在同一状态锁下处理并持久化，不等上游模型响应完成。这只保证策略/绑定账本原子性，不声称现有 spawn 工作区准备等外部副作用可回滚。
- **生命周期与通知独立**：已确认 `proposer_terminal`、策略被更新或提案被替换仍使 pending 作废；缺工具结果或通知失败不是生命周期终止证据。取消/截止在提交前阻止变更；提交后的取消不回滚历史策略，运行结束的未决提案记为 `voided(run_ended)`（无法持久收尾则标未决/unknown），不得在截止后继续审批。

### 3.2 消息约定

全部通过原生 `send_message(to, message)`；relay 在 `prepare_response()` 按**消息首行**识别。首行不匹配任何前缀的消息是普通消息，不解析。

| 前缀（首行） | 发送者 → 收件人 | 正文 | relay 动作 |
|---|---|---|---|
| `ROLE-POLICY <slot>`（可选 `base=v<k>`） | master → 该槽当前实例 | 第二行起为完整新策略 | base 取该 master 请求快照里的目标槽版本；仍为当前版本才记 base+1，provenance master_set；旧基准的 pending 作废 `policy_superseded` |
| `POLICY-PROPOSAL`（可选 `base=v<k>`） | worker → `leader` | `CHANGE:` 完整新正文；`REASON:`；`EVIDENCE:` 引用事件/任务/文件 | 槽、实例、基准取自本请求的注入记录；基准仍等于当前版本才处理；无 pending 则新建，同文补发，异文替换（见下） |
| `POLICY-DECISION #<id> approve\|reject\|amend` | master → 提案者实例 | approve/reject 可带 `NOTE:`；amend 第二行起直接是完整新策略，不嵌套前缀 | 槽与基准从指定提案取；提案须 pending、提案者 active、base==当前版本；approve/amend 记新版本，reject 只改状态；旧编号不重定向 |
| `GLOBAL-POLICY`（可选 `base=v<k>`，H-adaptive） | master → `"*"`，必须广播 | 第二行起为完整新 C | base 取该 master 请求快照里的全局版本；仍为当前版本才记 base+1；所有角色下一新请求注入，广播结果逐收件人记录 |

- **身份与越权**：发送者身份来自 `caller()`。worker 发 ROLE-POLICY/DECISION/GLOBAL、master 发 PROPOSAL、ROLE-POLICY 不发给该槽当前实例、PROPOSAL 不发给 leader、DECISION 不发给 proposer，一律拒绝。只有 master 的 GLOBAL-POLICY 可且必须 `to: "*"`；其余控制前缀仍禁止广播，普通广播不受此规则改变。拒绝走 `rejected_tool_response()`，稳定错误码包括 `stale_version`、`stale_proposal`、`unknown_slot`、`not_authorized`、`malformed_header`、`duplicate_name`；已有 pending 本身不再是拒绝原因。
- **初检只暂存，返回前才提交**：`prepare_response()` 校验该响应的**全部**调用，在临时状态按顺序模拟，返回待提交变更，不写策略/绑定账本。之后完成脱敏、model views、响应文件、attempt 记录和 usage 写入，包括 `handle()` finally 的最后一次 usage 写入。仅在这些本地操作都成功、响应仍可返回时，于 `handle()` 收尾且释放该角色 active request 之前，在同一锁内重查取消/截止、生命周期、绑定、pending 和快照 base，重新模拟并持久提交整批；随后发布缓存、释放请求并返回。不跨模型网络请求持锁。
- **晚提交失败的语义**：晚复核失败返回现有整响应工具拒绝；记录最终返回的拒绝响应、model views 和拒绝原因，保留 provider attempt 的原始结果，不将暂存的成功响应当成 CLI 最终响应。证据写入失败按本地 evidence failure 处理，两者均不提交策略、不触发 provider 自动重试。策略批次持久化失败也不发布状态；成功提交后不再安排会把此次成功改成失败的必需本地证据写入。HTTP 写回仍在此之后，断连风险保留，不承诺 relay 提交与 CLI 接收原子一致。
- **整批范围不变**：任一调用非法使整批失败；同响应两条提案按同文/异文规则模拟，set 后紧跟对旧基准 approve、重复 `tool_call_id` 等均拒绝。这是普通锁内的状态事务，不是等待原生消息确认的跨系统协议；不将策略原子性扩张为普通工具、spawn 工作区等外部副作用的回滚保证。
- **relay 提交即生效，适用于全部四种前缀**：PROPOSAL 成为 pending，ROLE/GLOBAL 生成当前版本，DECISION 完成批准/拒绝/修订，都以持久提交为准。之后即使响应写回断连、原生守卫拦下消息、通知失败或发送工具结果缺失，也不回退已经提交的状态。双方以其下一次注入的状态为准；这一点写进运行时披露，避免把原生发送错误理解为策略未提交。它不代表同一响应中的普通工具已执行。
- **通知结果单独记录**：用新增 `source_request` 与已有 `tool_call_id` 匹配原生工具结果。单播成功记 `sent`，明确错误记 `send_failed`，缺结果记 `unknown`；后续出现结果可追加。`client_disconnected` 仅说明连接异常，不证明工具未执行；原生发送成功不证明接收者已处理。结果和生命周期作废互不替代。
- **全局广播不因部分失败回退**：复用原生广播，在实际枚举/结算处加 compat 元数据，记录发送者、工具调用关联、实际收件人、各自 accepted/failed/unknown；不记录消息正文、不另建路由。发送集合以原生发送时成员表为准，不拿提交时的槽列表猜成功名单；terminal 成员仍可能在其中。记录 `total=0` 为零收件人，不算通知所有人成功。原生总体结果只供交叉核对；若缺逐项元数据或唯一调用关联，则相应收件人/结果为 unknown，不能用“total 减失败数”捏造具体成功者。accepted 指排队/写入成功，不是进入模型请求；queued/drained/observed 仍各自计量。未来首次加入的 worker 只读最新注入，不补通知。
- **发送报错与入队证据可以并存**：原生向 worker 发送时先入队、再 emit/必要时 flush；后续步骤失败可令广播某项为 failed，但消息已经入队。failed 只表示发送调用报错，不证明从未 queued 或永不 observed；若有后续投递证据同时保留，不互相覆盖、不回滚策略，也不因此自动补发。
- **三个写入前缀统一可选 base**：ROLE-POLICY、GLOBAL-POLICY、POLICY-PROPOSAL 的 base 都来自生成响应的固定请求快照；分别取目标槽、全局 C、本实例槽的版本。省略合法；提供时须与快照相等，提交时快照版本还须等于当前版本，否则 `stale_version`。缺快照/目标不得猜当前最新值或自动重基。DECISION 仍引用确定的提案及其 base，不另要求版本参数；spawn 初版/继任仍按绑定规则处理。
- **补发与替换共用一条规则**：在上述检查通过后，无 pending 则生成新编号；同一实例、同一基准下，解析后的 `CHANGE/REASON/EVIDENCE` 三个文本字段逐字相同则沿用 pending 编号、只追加重发通知记录，不新增提案或重置提交时间；任一字段不同则在同一批次把旧提案记为 `voided(replaced)`、新提案设为 pending。比较不做语义猜测、不新增哈希。不以 master 是否已收到注入为分界；每槽仍最多一个 pending。
- **决定引用确定的提案，不跟随替换**：approve/reject/amend 与替换在同一锁内排序。替换先提交，旧编号的决定返回 `stale_proposal` 及当前 pending 编号（没有则为 null）；批准先提交，旧基准下的替换返回 `stale_version`。错误里返回当前编号只是诊断，绝不把对旧提案的决定移到新提案；master 等下一次完整注入后重新决定。
- **提案消息不改写**：relay 不把提案编号塞进首行，不需要为此添加历史还原逻辑。master 从 system 注入读取编号、正文和状态；worker 从自己的注入读取编号与决定。迟到的原生通知可能描述已替换内容，披露明确以当前注入为准，不凭通知猜编号或恢复旧状态。
- **原生投递仅作通知与尝试唤醒**：消息照常走 worker 队列或 leader inbox/callback 路径，不新增独立唤醒通道。注入本身不会启动模型请求；通知失败或去向未知时，空闲 master 要等其他消息或 check-in，不能保证立即审批。没有后续请求时，记录 pending 未被携带，而不是伪造送达或自动补调用。
- **拒绝是整响应拒绝**（现有语义）：畸形输入以及替换/决定的良性竞态，都可能使同响应里无关的协调调用也不执行。记录被拒响应、被连带丢弃调用数与后续纠错成本（不可唯一归因则 unknown）；披露建议策略控制调用单独成一个响应，不新增强制拆分规则。
- **spawn 标记**：`agent.prompt` 里以一行 `ROLE-POLICY:` 开始、到 `END-ROLE-POLICY` 或 prompt 末尾的块记为该槽 v1（provenance spawn_initial，状态 planned）；一行 `SUCCEEDS: <name>` 触发继任。块保留在 prompt 里，worker 照常看到；relay 不改写 prompt 内容，只在末尾追加已有的那句说明（改为同时提到角色策略在 system 里）。
- **模式 off**（静态 H）：relay 不解析任何前缀，不注入，不记录；spawn 标记也当普通文本。

### 3.3 注入

每个新的 coding 逻辑请求采样一次策略视图；同一逻辑请求的传输重试复用该视图。worker 在 C 之后收到：

```text
Role policy: slot=decoder, instance=decoder2, version=3
Committed by master at 2026-10-10T12:18:00Z.
<approved text>
For your communication choices, this approved role policy overrides conflicting defaults in C or
incidental assignment instructions. It does not override tool permissions, safety boundaries,
cancellation, or the task deadline.
[R-propose] Your proposal #7: pending (base version 3).
Submitted at 2026-10-10T12:24:00Z; this is the fixed submission time, not its current age.
Continue working under the current approved policy while this proposal is pending.
Native message delivery is only a notification; the policy and proposal state shown here is authoritative.
```

master 收到：每槽的当前版本与**完整正文**（≤2000 字符/槽，不设固定槽数上限，记录实际注入长度），以及每条 pending 的 id、slot、proposer、提交时间、base 和**完整 CHANGE/REASON/EVIDENCE**（合计 ≤2000 字符/提案，每槽最多一条）。不能只给摘要、链接或编号再等原生消息补正文。每条内容关联本次逻辑请求；compaction 后仍能由 relay 状态重建。worker 的注入提供自己当前 pending（若有）和最近提案的决定/作废原因、被替换时的新编号，避免通知丢失后只能猜状态。

**确定性表示**：新策略/提案注入块是模式、接收角色及其可见持久状态的纯函数，槽与提案按稳定键排序，空白/序列化固定。无状态变化则文本逐字相同；不放当前时间、距今时长、请求号、attempt 号、采样次数或通知重发次数。时间只显示不可变的提交/决定时间；此前 `pending since 1.4h` 若使用只能表示固定提交偏移，本版改用明确 UTC 时刻。`source_request` 和采样时间留在证据中，不渲染到提示。生命周期/策略/提案状态真正改变时允许文本改变；不保证原生历史、已有预算提示等其他部分不变。

**注入与请求发送分开计量**：`payload()` 构造发生在部分截止/预算检查之前（`relay.py` `handle()`），所以 payload 带了提案不等于请求已经发出。注入记录列出请求快照中的策略版本和提案编号，再关联现有每次 attempt 的 `upstream_started`、结果和 usage。首次携带完整提案且 `upstream_started=true` 只称"首次携带提案发起上游请求"，不称 master 已读、理解或批准；启动尝试也不独立证明 provider 收到了完整请求。只有构造 payload、随后被预算/取消/截止拒绝的请求不计入这个首次发送指标。

**在途请求与通知不同步**：master 发起请求后才提交的新提案，不追加进该请求或同一逻辑请求的重试视图，只能等下一次新逻辑请求。原生通知即使更早进入 inbox，也不改变快照。注入不唤醒 master，不延长截止，不自动调用模型。

C 的注入：`global/master/propose` 三种模式下，master 的 C 不再写进静态 `--system-prompt`，改由 relay 每请求注入并标版本号（`Communication policy (version k)`），与 worker 同一机制；`off` 模式保持现状。

保证的只是"每次请求重新提供正文"，不是模型必然照做或立即收到；更新只影响后续新请求，在途请求保留旧版本关联。

### 3.3.1 前缀缓存与成本

`source_boundary.py:18–26` 的 `append_system_text()` 将文本加在第一条 system 末尾（不存在 system 时插到开头）。策略/提案视图因此位于后续历史之前；状态改变可能使**首个变化位置之后**的前缀无法复用，但之前不变的部分仍可能命中，具体取决于 provider 的缓存粒度、路由、保留时间等。不能写成“每次更新整段缓存必然失效”。master 的视图包含所有槽与 pending，受其他角色状态变化影响，worker 主要受自身可见状态变化影响。

本地旧 Optuna 运行 `runs/stage1_windows_15tasks_8g_20260916/run/train/batch_002/replay/candidate_001/task_009_opt-4.0.0-roadmap/team/relay/usage.json` 记录 prompt 99,485,733、cached 95,343,232（95.84%）。这里只证明旧运行曾依赖缓存；不是 FFmpeg/Kimi 数据，更不是本方案的动态注入效果。本次未连接服务器。

按臂报告：总已知 prompt tokens、缓存字段报告覆盖（attempt 数及 prompt-token 覆盖）、已报告部分的 cached/uncached tokens 与 `sum(cached)/sum(prompt)`，同时列失败/未知 usage。现有 `usage_and_cost()` 会把缺缓存字段变成 0，新增投影必须在归一化前保留“是否报告缓存指标”；`usage_complete=true` 不代表缓存数据完整。缺字段为 unknown，不能纳入零命中分母；没有价格数据不估费用。不以历史 95.84% 推算新臂成本。

S0 先用脚本模型验证状态不变时的文本稳定性、状态改变的范围及缓存字段缺失处理。另经授权的真实 provider 短跑中，做不变状态连续请求以观察预热，再切换一次策略状态并重复请求；记录各次 prompt/cached/延迟/路由可见信息，区分普通历史增长与策略变化。必须确认 provider 报告缓存字段；未报告则结论是不可观测。该测量用于成本诊断，不把必定命中或必定下降当通过条件，不另引入缓存率门槛。

### 3.4 与原生提示、A/C 的关系

- 前缀约定的说明放在运行时披露里（`runtime_rules()` 同位置），按臂只给对应角色：R-master 的 master 看到 `ROLE-POLICY` 与 spawn 标记说明；R-propose 的 worker 另看到 `POLICY-PROPOSAL`、master 另看到 `POLICY-DECISION`；H-adaptive 的 master 看到 `GLOBAL-POLICY`。说明里写清何时适合提议、仅提自己的发送规则、待决期间继续按现行策略工作、注入为权威状态、通知错误不等于策略未提交、旧编号决定不会转移到替代提案。
- 四种模式的**实际英文披露草稿及字符数**见 §9，S0 使用这些文本并记录最终实际注入长度，不以本节要点清单代替提示。D10 未定，两个备选尾注分列，不因本次落档自动选择。
- 起始 A/C 文件不人工改写。`role_policy_mode ∈ {off, global, master, propose}` 通过现有共享 runner 配置路径传入，运行清单记录请求值与实际生效值。
- outer optimizer 本轮不改；各臂起始 A/C 固定。H-adaptive 的 C 在线变化是明确的实验变量，R-master/R-propose 的角色策略在线变化同理；不把“起点固定”误写成在线对象不可变化。

### 3.5 安全边界（如实陈述）

v3 的策略入口与现有工具过滤、预算同处一个信任层：相信 compat 层上报的 owner 头与系统提示一致，而这两者都在共享的运行凭据之下，不是角色级强认证。新增的策略写入是**新的权限敏感行为**：它让一条模型请求的响应能改变别的角色之后看到的指令。worker 用 shell 拿 argv 里的 relay token 伪造一条 master 的 chat 请求、再让模型生成策略消息，理论上可行；这与伪造请求去花预算属于同一类蓄意行为，实验采用的是"信任原生调用路径、不防同权限攻击者"的受限威胁模型。现有网络、文件、工具约束全部保留。

### 3.6 事件与投影（schema 3 → 4）

| 阶段 | 事件 | 能证明什么 |
|---|---|---|
| 状态提交 | `role_policy_committed` / `global_policy_committed` / `role_policy_proposed` / `role_policy_decided`（含新增 `source_request` 关联字段、已有 `tool_call_id`） | `handle()` 收尾晚复核及批次持久化完成，策略或提案状态已生效；初次暂存不算提交，不说明原生工具已执行 |
| 提案作废 | `role_policy_proposal_voided`（原因：`proposer_terminal`、`policy_superseded`、`replaced`、`run_ended`；替换时含新编号） | 生命周期、版本或替换使提案失效；不由通知结果触发 |
| 同文补发 | `role_policy_proposal_renotified`（已有提案号、新通知的 `source_request`、`tool_call_id`） | 重复通知意图已记录，没有新建提案、重置时间或确认消息发出 |
| 通知结果 | `role_policy_message_sent` / `role_policy_message_send_failed`；缺工具结果的通知状态为 `unknown` | 原生工具结果已确认的发送结果；与策略状态无关，缺确认不是失败 |
| 广播通知 | compat 在原生广播枚举/结算点记录收件人、各自 accepted/failed/unknown、总数与可唯一关联的工具调用；relay 投影按收件人展开 | 实际发送集合及原生排队/写入结果；零收件人、terminal、部分失败单列，不把总体成功当全部 observed；无唯一关联只保留聚合/unknown |
| 绑定 | `role_policy_bound`（planned→active）/ `role_policy_binding_ended`（来源：stderr 增量读取的 `worker_terminal`，记消费延迟） | 槽与实例关系 |
| 拒绝 | `role_policy_message_rejected`（初检/晚复核、错误码、发送者、目标、该响应被丢弃的其他调用数） | 哪条消息没进状态，以及整响应拒绝的连带范围；持久化失败另记 evidence failure |
| payload 注入 | `role_policy_injected`（实例、槽、策略版本、提案编号、注入字符数；准入后关联 `source_request`，否则 unadmitted） | 请求快照已包含相应完整正文，不说明上游调用启动；关联字段不进入提示文本 |
| 请求尝试 | 新增 `source_request` 关联现有 `attempt/retry_index/upstream_started`、HTTP/错误状态与 usage；retry 沿用逻辑号和快照 | 哪次上游尝试携带该版本/提案；不证明远端接收或理解，不另造 `observed_by_master` |
| 缓存与延迟 | 在 usage 归一化前提取缓存字段及 presence；按臂汇总 prompt/cached/uncached、覆盖率和请求延迟，并关联可见状态变化 | 缓存实测及缺失，不将缺字段变为零、不单凭状态变化推断 cache miss；失败尝试和未知用量保留 |
| 原生通知（→worker） | 复用现有 `team_message_queued/observed`，可唯一关联时记录 `policy_kind` 与消息身份 | 原生队列/进入请求的阶段，与 relay 注入分列 |
| 原生通知（→leader） | 可确认的提案通知信封进入 master 请求时独立记录；缺唯一关联记 unknown | 消息进入 native 历史，不证明此次请求是由该通知唤醒；不充当提案有效性的条件 |
| 行为 | 复用协作时间线与消息记录，按生成请求的版本关联 | 版本切换前后的行为描述 |

- **审批与通知是两条时间线**。提案路径统计：提交→首次携带完整提案发起 master 上游请求→决定提交；批准/修订后另统计新策略提交→首次携带该版本发起 worker 上游请求。通知工具结果可能晚于决定，不强行插入审批阶段，也不据此生成负的"审批等待"。没有后续请求/决定时保留未发生、等待截至采样时间或运行结束；缺证据记 unknown，已确认替换/终止按对应原因单列。
- 投影新增 `role_policies` 段：每槽版本数、payload 注入覆盖、首次携带策略/提案发起上游请求的延迟、提案数/决定率/批准率/作废率（按原因）、同文补发次数、替换次数、stale 决定次数、通知 sent/send_failed/unknown、按请求实际版本关联的该角色消息数与收件人分布（描述性）。同文补发不增加提案数，异文替换的新编号计一个新提案；替换提案不从分母剔除。usage 缺失和请求失败另列，不当成成功处理或零成本。
- 广播按实际收件人展开统计发送结果，和定向通知按同口径报告；另列广播调用次数、部分失败和零收件人，不能把一条广播直接当一次成功送达。按臂关联缓存字段覆盖与命中率（§3.3.1）；拒绝关联被丢弃的非策略协调调用数、纠错请求数/已知 token，可唯一归因才归到该次冲突，否则只给整体纠错成本或 unknown。
- 去掉消息编号改写后，旧版本、被替换内容或同文补发的通知可能无法唯一关联到一次提案/通知；保留通知层计数与关联 unknown。没有原生因果记录就不计算逐提案精确唤醒成功率，不能从 sent、信封出现或 system 注入推断"这条消息唤醒了 master"。
- 策略消息在通信指标里单列 `policy_control`，不计入横向/任务通信。
- 旧运行标 `not_instrumented`，新 H 标 `off`，启用未使用记零提交与完整覆盖。缺关联、缺 usage 记 unknown。
- 正文、理由、证据引用沿用既有脱敏与训练证据边界。

---

## 4. 运行时改进：独立工作项

角色机制必需的运行时接入只有 §3 的识别、注入与绑定。以下各项独立验收；纳入正式实验时三臂统一。

**R1 轮询治理**。`task_list` 本有结果感知保护；10-02 的 318 次是与状态读交替，连续相同守卫抓不到；`read_file` 第五次在执行前被拦，所以"工具内等待"绕不过守卫。顺序：先记录 master 绝对调用量、token、等待时间、终止原因并完善披露；对确实缺通知的任务板变化评估合并后的 native callback 通知（去重显式/自动汇报）；若仍需抑制重复状态读，单独设计位于守卫边界的"结束 master 回合→原生等待"，仅单一只读调用、无其他待执行调用时适用。成本改善由真实模型另测。

**R2 provider 故障分类与恢复**。`relay.handle()` 已在同一 native 请求内有界重试，受硬截止约束。可重试传输故障、鉴权/配置错误、逻辑期限耗尽、用户取消/硬截止四类分开处理；跨请求恢复若做，要定义真实的 `running→suspended→resumed/terminal` 原生入口，不重放已执行工具、不拼接 attempt 输出、不伪造 tool result。角色槽继任独立于会话恢复，不自动复活。

**R4 compaction**。先核对实际上下文上限、原生触发路径与 usage，测摘要丢失、注入重建、失败/取消，再定阈值；三臂统一。

**R5 时间规则**。披露 check-in 时间表、刷新间隔与非精确唤醒；策略要求的时间点若无对应唤醒，记录实际执行时间与偏差。**不加 validator**，代价是 B7 那类"为凑 70%/85% 检查点而轮询"的成本不会被阻止，只被测量；定时唤醒工具列为后置。

**R6 交付证据**。commit 数、工作区文件变化数、功能进展三者分别保留；文件级计数用已有初始工作区来源。check-in 快照只叫运行状态摘要，不当交接。

**R7 评分**。功能层为主结果；各层映射按官方测试身份核对后汇总；保留总分、各层、执行故障、覆盖率。pandoc/ctags 只是候选。

后置：共享类型化上下文、自建计分板、按回合调用观测、定时唤醒工具。

---

## 5. 实施顺序与验证

### 5.1 分段

| 段 | 交付 | 验证证据 | 范围 |
|---|---|---|---|
| S0 | 采用 §9 四模式披露草稿、标注实际长度；前缀识别/拒绝原型与确定性注入 | 脚本 provider 下合法/畸形/越权/stale 全路径、广播/晚提交故障、稳定文本与缓存字段缺失；另经授权短跑观察 master/worker 遵从率及一次状态变化前后的缓存/延迟 | 不把缓存必命中或必下降设为通过条件；遵从率不可接受则评估 compat 原生工具 |
| S1 | `role_policy.py`：槽/实例/版本/提案/全局版本、锁与整响应事件批次、同文补发/异文替换 | 单元 + 并发 + 整批事件重放；无通知确认依赖 | 不接实验 |
| S2 | relay 四前缀/spawn、初检暂存与 handle 晚提交、新增逻辑请求关联、逐收件人广播元数据、stderr 增量读取、确定性完整注入（含 master 的 C）、披露；D10 判断（若采纳） | 真 CLI 脚本场景：set→inject、提案→master 全文请求→决定→worker 新版；迟到/失败/未知通知、晚冲突/证据失败、继任、compaction 重建、最终汇报副作用 | 默认 off，不改写提案消息，不改广播路由 |
| S3 | schema 4 投影、缓存 presence/覆盖、拒绝连带成本 + 配置传递 | 合成竞态轨迹、缺缓存字段、部分广播与历史离线兼容 | 不覆盖旧结果、不把缺失计零 |
| S4 | R1/R2/R4/R6 独立交付 | 各自复现测试 | 不阻塞 S0–S3 |
| S5 | 实验配置与短跑/正式比较 | 同期三臂、全样本、成本与故障分列 | 真实模型、部署、运行需单独授权 |

### 5.2 必测场景

| 组 | 场景 | 应验证 |
|---|---|---|
| 识别 | 四种前缀合法形式；ROLE/GLOBAL/PROPOSAL 均省略/提供 base；amend 直接附完整新策略；首行不匹配、大小写/空格变体、正文超长 | 仅按定义首行解析；三类缺 base 均合法，显式值核对快照；amend 不嵌套；正文有界 |
| 越权 | worker 发 ROLE/DECISION/GLOBAL；master 发 PROPOSAL；ROLE 发给非当前实例；PROPOSAL 非 leader；DECISION 非 proposer；前三种控制消息广播；GLOBAL 单播 | 均拒绝、不写状态；唯一合法控制广播是 global 模式下 master 的 GLOBAL-POLICY |
| 整批提交 | 合法 ROLE + 同响应非法调用；同文/异文 PROPOSAL；set 后 approve 旧基准；重复 `tool_call_id`；事件持久化失败与重放 | 初检只暂存、晚复核后整批持久化；失败无部分策略状态；不承诺普通工具外部副作用回滚 |
| 晚提交边界 | 初检后分别在脱敏、model views、响应文件、attempt 写入、handle 最终 usage 写入失败；初检后发生替换/terminal/set/取消/截止；提交后 HTTP 断连 | 所有前置本地失败均不提交；同锁晚复核拒绝旧状态，最终拒绝证据/历史一致且不自动重试；提交后断连不回退 |
| 全局广播 | master GLOBAL 发 `*`；集合改变、terminal、零收件人、部分/全部失败、入队后 emit/flush 报错、缺埋点/关联、之后新 worker 加入 | 提交独立有效；按实际枚举逐收件人结果，不从总体成功推断 observed；failed 与 queued/observed 可并存；新 worker 注入最新 C，无补广播 |
| 通知与状态分离 | 四种操作提交后分别出现 sent / Failed / Error / 无工具结果；后续迟到结果；原生守卫拦下消息；响应写回断连 | 已提交状态不回退；缺结果只记 unknown；发送结果不替代生命周期；普通工具不因策略提交被标为已执行 |
| 发送早于确认 | worker 同响应发送提案并运行长工具；master 在发送者下一次请求前得到提案注入并批准 | pending 无需等待工具结果；决定可提交；迟到发送确认不改变决定 |
| 通知无确认与终止 | 提案发出后 worker 没有下一次请求，分别模拟仅断连、守卫结束回合、真实 terminal、硬截止 | 前两者不凭缺结果作废；真实 terminal/run ended 按生命周期处理；截止后无新审批 |
| 防止迟到回退 | ROLE/GLOBAL/DECISION 生成 v2，后续已生成或注入 v3，v2 通知再报告失败 | v3 仍是当前版本；历史版本与请求归属不改写 |
| 补发与替换 | master 携带过/未携带过提案时分别重发同文/异文；同 CHANGE 但 REASON/EVIDENCE 改变 | 同文沿用编号和原提交时间，仅增通知；任一正文段变化都替换旧编号；不按 master 是否见过分支 |
| 版本与替换竞态 | 三个可选 base 前缀分别测显式不符、快照过期、缺快照/目标；set/terminal 与 approve；替换与 approve 两种顺序；旧编号 reject/amend 与无关协调调用同响应 | 不猜最新 base；同锁定序；旧决定不转移；整个响应被拒及连带丢弃调用均记录，不当模型能力失败 |
| 绑定 | spawn 带块 → 首次请求 active；spawn 失败 → planned 作废；重名 spawn；SUCCEEDS 指向存活前任；SUCCEEDS 指向 relay 尚未消费到 terminal 的前任；两次并发 SUCCEEDS；块与 SUCCEEDS 同传 | 槽不分叉、版本不重置、pending 作废、失败不留绑定；重名与未确认终止都被拒绝 |
| 生命周期输入 | 首次请求绑定；stderr 延迟、重复行、半行、CLI 退出尾部；另有长模型请求在途 | 入口绑定/terminal 独立在状态锁内处理，不等模型响应晚提交；消费延迟实测；去重；无丢失或丢失可计数 |
| master 提案输入 | master 下一新请求、compaction 后请求、已有在途请求的 retry；迟到的旧提案通知 | 新请求有 pending 编号和完整正文；重试沿用旧快照；不靠信封补正文、不改写提案首行 |
| 最终汇报副作用 | worker 发 PROPOSAL 后继续工作、回合末输出文本；发消息后以无正文响应结束 | 已测：非空正文照常转发；无正文的结束响应被原生重新请求（§10.1）；补丁开/关的直接对照尚未做 |
| 注入与上游尝试 | payload 构造后预算拒绝/取消/截止；attempt 启动后失败；compaction 后；同请求 retry；长请求期间策略改变 | payload 记录不算上游已发起；按实际 attempt/结果计量；retry 不换版本；旧版动作不归新版本，不称模型已读 |
| 逻辑请求号 | 无调用上限、有调用上限、跨角色并发、同请求 retry、未准入 payload | 捕获已有逻辑计数得到 run 内唯一 `source_request`；所有 retry 共用；无预算也关联；未准入不编造号 |
| 注入确定性 | 固定状态连续构造、时钟推进、同文 renotify、字典顺序不同、单槽状态变化、compaction | 角色可见状态不变则策略块逐字一致；无动态时钟/计数污染；稳定排序；变化限定在可见状态；不声称全 payload 不变 |
| 缓存计量 | `prompt_cache_hit_tokens` / `prompt_tokens_details.cached_tokens` 显式零、非零、缺失；usage 完整但 cache 未报告；失败 attempt | presence 在归一化前保留；按覆盖的 prompt 分母算命中率；缺字段 unknown；另授权真实预热→变化→重复的诊断，不用固定命中阈值 |
| 唤醒 | master idle/busy 时通知成功/失败/未知；决定后 worker idle/busy；等待其他消息/check-in；用户取消与硬截止 | 注入不会唤醒；通知失败后 pending 可有效但未被后续请求携带；不补调用、不续期 |
| 模式与披露 | §9 off/global/master/propose 按角色组装、计字符；D10 两种备选分别测；固定披露和动态块分别计长 | off 新增文本长度为 0；只显示本角色入口，master 模式 worker 不见 PROPOSAL 语法；D10 不误称已实现；草稿长度不冒充 token 数 |
| 投影 | 策略消息单列；通知确认晚于决定；同文重发/替换/缺唯一关联；旧运行 not_instrumented；启用未使用 | 审批与通知独立；同文不增加提案分母，替换不剔除旧提案；unknown 不补失败或零延迟，不制造精确唤醒率 |
| 研究披露 | worker 提案 pending 时继续任务、按当前策略发消息；希望其他角色改变规则；批准后实际行为不遵从 | 文本明确待决不必停工、正式提案只改本角色发送规则；普通建议单列；批准不等于权限开通或行为遵从 |

本地 CLI 测试通过后，在另行授权的 Linux 官方容器跑相同场景。真实模型短跑只用于看遵从率、策略响应与成本；不使用约定是研究结果，不是剔除理由。

### 5.3 文件边界

新逻辑放 `agent_company/policy/role_policy.py`；`relay.py`（`prepare_response` 暂存、`handle` 晚提交、`payload`、请求关联和 cache presence）、`qwen.py`（披露文本、模式参数）、`qwen_native_compat.mjs`（广播元数据；D10 若采纳）、`programbench_comparison.py`（臂字段）、`programbench_evidence_projection.py`（schema 4）最小接入。不碰另一会话的 `programbench_study.py`、`run_programbench_evolved_followup.py` 及相关测试。提交、push、部署、付费实验都不在本次修订授权内。

---

## 6. 风险

- **遵从率**：模型可能写错前缀、把可选 base 写成与请求快照不符的值、把策略写进普通消息。ROLE/GLOBAL/PROPOSAL 省略 base 都不是错误。S0 使用 §9 草稿直接测；纠错占用调用与 token。
- **整响应拒绝的连带**：不只畸形输入，替换/决定间正常并发也会让同响应其他协调调用不执行；晚复核会捕获初检之后才发生的冲突。需重新生成协调动作，可能增加延迟和 token。披露建议单独发策略控制响应，计量连带丢弃与纠错成本，不把良性竞态一概归为模型失误。
- **提交后通知失败**：发送者可能看到原生消息错误，而策略已经持久提交。双方后续注入提供权威状态，不能因这个错误重放决定、回退版本或重新执行普通工具。若没有后续请求，状态虽有效但可能未影响行为，必须如实记录。
- **等待与替换代价**：没有通知就没有额外唤醒保证；审批可能等到其他消息/check-in。允许替换已被 master 请求携带的提案，会产生 stale 决定；持续修改可能让审批迟迟无法完成。记录替换数、失效决定和未决时长，不新增"已看过不可替换"的冻结规则。
- **广播不等于全员已读**：原生枚举含 terminal，成员可能在提交与发送之间变化；部分失败或零收件人的总体工具结果也可能显示成功。逐收件人结果与后续 observed 分开；若无法唯一关联，则 unknown，不为追求“对等”新增唤醒路径。
- **缓存与臂间混杂**：动态 system 前缀可能降低缓存复用、增加成本与延迟；master 汇总视图对任一槽变化敏感。确定性渲染只消除无意义变化，不消除真实状态更新的缓存代价。缓存 presence 单独审计；旧 Optuna 的高命中率不代表本轮 provider。不同臂更新频率/长度带来的代价属于机制包，不能仅归因于通信内容。
- **注入与唤醒的可观测边界**：payload 构造、上游尝试启动、通知发送、信封出现分别说明不同事实；没有唯一消息关联或原生因果记录时，逐提案唤醒情况为 unknown。注入与批准都不证明模型理解或遵从。
- **stderr 增量读取**：依赖 Node 对管道的 stderr 写入与宿主机文件刷新，延迟未测；丢行或乱序会让 terminal 延迟确认，影响 `SUCCEEDS` 与 pending 作废的时机。`SUCCEEDS` 在未确认时保守拒绝；审批仍按 relay 已消费到的生命周期判断，终止事件尚未消费时可能先提交决定，因此记录消费延迟与实际事件次序，不声称这种延迟对所有操作都无影响。
- **提案与自动汇报**：已测场景中提案不影响非空回合末正文的转发（§10.1）；模型持续给出空响应时的情形未测到。
- **提案风暴**：每槽一个 pending 限制了同时待决的数量，不限制累计补发或替换次数；master 仍要花回合处理。计数与延迟按 §3.6 记录。
- **自然语言冲突**：优先级声明不能保证模型解决所有语义冲突；冲突案例作错误分析。
- **继任≠恢复**：继承的只是策略正文。
- **效应与噪声**：同题少量运行只能给有限证据；横向比例或前后窗口变化不构成因果。

---

## 7. 决定点展开

- **D1**：批准的角色通信条款覆盖可变默认 C/临时指令，不覆盖固定运行边界。
- **D2**：已定 `send_message` 约定。理由：发送者身份与现有工具过滤同层，不需要 MCP 跳上的认证；原生消息复用既有通知/唤醒路径，但不保证及时唤醒或零 token 成本；不改写提案消息。全局更新必须广播，compat 仅新增逐收件人元数据；D10 判断仍待定，升级路径是 compat 原生工具。
- **D3**：首轮 H-adaptive / R-master / R-propose；静态 H 可选。
- **D4**：`SUCCEEDS: <name>` 仅对已终止前任；版本延续，pending 作废。
- **D5**：声明优先级并记录行为；原生自动汇报如实计量。
- **D6**：先做最小机制；R1/R2/R4 分开。
- **D7**：分段，每段可审核；真实模型与正式对照另行授权。
- **D9**：H-adaptive 作基线，前提是 §2 的 master C 注入、全局更新/广播通知、spawn 初版不对称计数。缓存命中/缺失、广播实际覆盖与披露长度也按臂报告；不称修改机会或唤醒已经完全对等。若只要“有无角色策略”对照，静态 H 可加回。
- **D10**：建议 compat 一行判断（首行为 `POLICY-PROPOSAL` 的消息不进 `explicitLeaderReports`），放在已有补丁模块里，用现有锚点方式与 drift 测试保护。替代是只披露"发过提案的回合要自己发最终报告"，代价是 R-propose 臂多一条行为要求。

## 8. 版本修订记录

**v3.4（2026-10-13，待审核）**：§9 新增两段披露（修改依据 Q、审批复查 V），细化 P-M / P-W 的字段要求，launch.json 记录披露版本；权限、消息格式、长度上限、解析器与 relay 行为都不变。详见 §12。

本节保留修订历史；旧版的确认后生效、消息改写、全禁广播与早提交规则不与正文并存。**当前规范以 v3.3 的 §0–§7 及 §9 披露草稿为准**；以下旧版本表格仅为历史。

| v2 内容 | v3 处理 |
|---|---|
| §3.4 MCP 调用链、MAC 信封、运行级密钥、调用局部上下文 | 删除。入口改为带身份一致性检查的模型请求里的 `send_message` 前缀（§3.2）；安全边界改为如实陈述（§3.5） |
| §3.5 五个 MCP 工具 | 删除。`get_*` 不需要：策略每请求在 system 里；set/propose/decide 变成三种前缀 |
| §3.2 operation_id 去重 | 删除。模型重发即新操作；只保留 base 版本检查 |
| §3.1 原生实例 ID、首次请求前的 native 接入点 | 改为"第一次 coding 请求即绑定"，由 relay 现有 `caller()` 承担 |
| §3.3 唤醒通道（adapter 调 `leaderMessageCallback`、补发核对） | 删除。提案是发给 leader 的原生消息，唤醒由原生投递完成 |
| §1.3 删除"7–22%" | 恢复为可用数据，要求拆"晚到/从未 observed" |
| S0 MCP 发现/认证原型 | 改为约定遵从性冒烟 |
| 5.2 HTTP 绕过、认证信封篡改场景 | 删除；保留越权（按发送者身份）场景 |
| §2 三臂 | 基线改为 H-adaptive（D9），新增 `GLOBAL-POLICY` 前缀与 `global` 模式 |
| 其余（状态语义、全样本、事件分阶段、R1–R7、文件边界） | 保留，文字压缩 |

### v3 → v3.1（10-10 审查）

| 审查问题 | v3.1 处理 |
|---|---|
| [P1] 提交后原生消息可能没发出 | §3.2 接受→发送确认两步；按 `tool_call_id` 匹配工具结果；失败/未确认作废；补发语义 |
| [P1] 单条提交与整响应拒绝冲突 | §3.2 两阶段提交：先校验整批、临时状态算变更、全部通过才写 |
| [P1] relay 无实时 terminal | §3.1 增量读取宿主机 `qwen_stderr.txt` 的生命周期行；未消费到即拒绝 `SUCCEEDS`；§5.2 延迟/去重测试 |
| [P1] worker→leader 无送达埋点 | §3.2 提案号改写进正文；§3.6 `observed_by_master` 以 master 请求里的带号信封为准；延迟分四段 |
| [P2] 提案压掉原生最终汇报 | §1.3 记录机制；D10 两个选项；§5.2 场景 |
| 措辞："已认证请求""不新增攻击面" | §1.3、§3.5 改为共享凭据下的一致性检查、受限威胁模型、新增权限敏感行为 |
| 措辞：H-adaptive 对等 | §2 三件事（master 的 C 改注入、全局更新路径、spawn 初版不对称计数）；§3.3 注入说明 |
| 新查到：原生同名 spawn 拒绝、relay 幂等注册 | §1.3 事实；§3.1 名字 run 内唯一、relay 先拒；错误码 `duplicate_name` |
| 新查到：`to: "*"` 广播 | 带前缀的广播拒绝；不用于全局策略 |

### v3.1 → v3.2（10-10 提案流程修订）

v3.1 把状态有效性依赖于发送者下一次请求的工具结果，会出现两个不同进程之间的次序问题：提案已到 master、但发送者还在等同响应的长工具，master 的决定早于发送确认；或提案已发送，却因发送者不再发起请求而被当成未确认作废。set/decide/global 同样可能在后续版本已生效后遭遇迟到失败回退。v3.2 删除这种依赖，不用更多确认状态去补偿。

| 修订项 | v3.2 当时规则（v3.3 的调整见下） |
|---|---|
| 接受→发送确认后生效，失败/未确认回退 | 全部四种前缀均以 relay 整批持久提交生效；原生通知结果独立，缺结果为 unknown，无发送驱动回退 |
| 整响应拒绝可能留状态 | 保留先校验/模拟整批、后持久化/更新缓存；用锁、事件批次与重放测试保证原子性，不新增跨系统确认协议 |
| 提案号改写与历史还原依赖 | 删除；master 注入编号及完整 CHANGE/REASON/EVIDENCE，worker 注入自己的提案状态 |
| 必填提案 base、amend 嵌套 ROLE-POLICY | base 从请求快照取，显式值可选且必须一致；amend 直接写完整新策略，槽取自提案 |
| 以 master 是否 observed 区分补发/替换 | 同文沿用编号，异文替换；先做生命周期/base 校验；旧编号决定拒绝且只提示当前编号，不重定向批准 |
| 作废条件 | 保留 terminal、policy_superseded、replaced 与 run_ended；发送失败/缺确认不作废，不以缺结果推断 terminal |
| observed_by_master 与串行四段延迟 | 区分 payload 注入、已有 upstream_started/attempt 记录与通知；审批时间线不等待通知确认，不称已读或精确唤醒 |
| 研究解释 | 正式认可的文本策略，不是通信权限开关；正式提案只覆盖本角色发送规则；待决期间继续按当前策略工作 |
| 其余边界 | 名字唯一、stderr 实时终止输入、D10、整批校验与实验决定点保留；本修订不授权实现或实验 |

核对依据：`relay.py` 的 `payload/caller/prepare_response/rejected_tool_response/restore_model_history`，`team.py` 的 `WorkerRegistry.register/contains`，`qwen.py` 的 `runtime_rules/master_system/native_lifecycle`/启动命令，`workspace.py` 的 `Popen` stderr 落盘，`qwen_native_compat.mjs` 的 `record()`、check-in 投递与 `team_message_*` 埋点位置，`programbench_evidence_projection.py` 的 `delivery_projection`；Qwen 0.23.1 bundle 的 `sendMessage()`（含 `classifyShutdownResponse`、`explicitLeaderReports`、leader inbox 分支）、`onStatusChange` 的 idle 分支、`generateUniqueTeammateName`、`formatLeaderEnvelope`、`send-message` chunk 的工具 schema 与结果文本、`EXCLUDED_TOOLS_FOR_TEAMMATES`。历史数字沿用已注明的报告，本次没有连接服务器或重读模型轨迹。

### v3.2 → v3.3（10-10 缓存、广播与提交边界审查）

| 审查项 | v3.3 处理 |
|---|---|
| 动态前缀缓存成本遗漏 | §2/§3.3.1/§3.6/§6：确定性状态块、固定提交时间、按臂报告缓存 presence/覆盖/命中/延迟；S0 另授权观察变化前后，不声称整段必失效 |
| GLOBAL 单播造成通知覆盖不对等 | GLOBAL 必须 master→`*`，其他控制前缀仍单播；记录实际枚举、零收件人、terminal、部分失败和 unknown；不回退策略、不保证全员唤醒 |
| prepare_response 提交过早 | 初检仅暂存；跨过 `_attempt()` 和 `handle()` 最后证据写入后，同锁晚复核并提交；保留断连语义，失败不触发传输重试 |
| 三种 base 规则不一 | ROLE/GLOBAL/PROPOSAL 全部可省略；显式值对请求快照，提交再对当前状态；缺快照不得猜值 |
| source_request 被误称已有 | 新增关联字段，取准入时已有逻辑计数；retry 共用，未准入不造号，无预算也记录；不加 UUID/operation_id |
| 良性竞态的连带代价遗漏 | 整响应拒绝同时丢弃其他协调调用；记录范围与可归因纠错成本，披露建议控制响应单独发 |
| 只有披露要点，没有实际文本 | §9 给出四模式按角色组装的英文草稿、实测字符数和 D10 两个备选；off 新增文本为零；S0 用实际文本而不是要点清单 |

本次额外只读核对：`source_boundary.py:18–26` 的注入位置；`relay.py:208–220` 的 cache 缺失默认零、`527–627` 的逻辑准入/重试/最终 usage、`741–806` 的校验后写入、`969–987` 的 HTTP 写回；Qwen `send-message-AWKUDJBH.js:284–323`、`chunk-PDUMGZ2P.js:81654–81671` 的广播范围/结果与 scheduler 的 callId 传递边界。09-16 Optuna 数字仅复核本地 `relay/usage.json` 聚合字段，不读取请求/响应或内部推理。当前文档检查不能替代将来的脚本模型、真实 CLI 和授权真实模型测量。

---

## 9. S0 使用的运行时披露草稿

以下是**待实现的英文模型输入**，不是已经部署的提示。为避免四份文本重复维护，列出固定公共块与模式/角色块，并明确逐字组装方法；组装后就是各臂 S0 的实际披露候选。不得把另一个角色的入口说明也注入给本角色。`off` 完全不加公共块。沿用其他既有 runtime/source-boundary 提示，不在这里复制。

各块仅取代码围栏内部内容，去掉块首尾换行，块之间固定用两个 LF（`\n\n`）连接；字段占位符是语法说明，不随请求替换。动态状态块单独放在固定披露之后，内容依 §3.3 确定性渲染，固定文字在同一模式/角色内不随策略更新变化。字符数包括空格与内部 LF，按 Unicode 字符计，不是 token 数；已有提示、动态状态正文与消息历史不计入下表，S0 另记录实际总长度/usage。

### 9.1 固定公共块 B（仅 global/master/propose）

```text
Policy state is maintained by the relay. The policy/proposal snapshot supplied in this request is authoritative; a native policy notification may be late or missing. Committed state is not rolled back if native message sending fails. Check the next supplied snapshot before retrying an update; do not treat a send error as proof that nothing changed.
Updates affect later new requests, not a request already in progress or its transport retries. Displayed timestamps are fixed submission/commit times, not the current time or an age. When idle, end your turn and wait for normal messages or check-ins; do not repeatedly poll for policy or proposal changes. Injection itself does not wake an agent or extend a deadline.
Use only the control-message formats allowed below for your role and mode. Put the exact header on the first line of send_message.message. Optional base=v<k> must match the supplied snapshot; the relay also checks that this base is still current. Missing state is not permission to guess a version. A rejected control call can reject every call in the same response, including unrelated coordination. Prefer a separate response for policy controls; after a stale rejection, use the next supplied snapshot and reconsider the operation.
These are communication instructions, not new tool permissions. Existing safety rules, source boundaries, cancellation, and the hard task deadline always apply. Ordinary task messages remain available.
```

### 9.1b 修改依据块 Q（v3.4；global/master/propose 的所有角色，紧跟 B）

```text
Changing communication policy: do it only for a concrete coordination problem you can point to (such as duplicate notices, an incomplete handoff, a message to the wrong owner, or an interface change that reached its users late) or because the task changed so the current rule no longer fits (a new dependency, the start of integration). Consider whether the rule is inadequate, was not followed, or a message has simply not arrived yet. If a rule was clearly not followed, remind first. A rule that already covers the case may still be simplified, clarified, or given a different trigger when a concrete problem shows it does not work in practice. If the cause is not established, state your hypothesis; you do not need to prove the root cause first. You usually cannot see whether a message was delivered or read; say so rather than assume. Fix one problem per change; "communicate better" is not a change. Cite evidence a teammate can locate (a message's sender and time, a task, a commit, a file, a check-in), and say what you expect to observe and at which related event it can be checked, such as the next handoff or check-in; do not poll for it. The same standard applies to ordinary policy suggestions.
```

### 9.1c 审批与复查块 V（v3.4；global/master/propose 的 master，放在该角色最后）

```text
Before you change a policy or decide a proposal, check four questions. Grounded: is the problem observed, or a guess, and if its cause is not established, is the hypothesis stated? On target: does the change address that problem? Remind first when an existing rule was clearly not followed, and do not approve a repeat of an existing rule without a substantive change; simplifying, clarifying, or retriggering a rule is a substantive change. Executable: do permissions, message delivery and wake-up conditions allow it? Worth it: what improvement could be observed, and could it add useless reports or cause important messages to be missed? Change only when all four hold, and explain a direct change or an amendment in a short ordinary message, not in the policy text.
Do not assume a committed change affected workers immediately. Review it against reports and task events that arrive naturally: when a related event occurs, record one conclusion in an ordinary message starting with POLICY-REVIEW <slot or C> v<k>: improved, no change, side effects, or cannot judge yet, plus the facts. If you cannot tell whether behavior followed the new version, write cannot judge yet; do not poll for it. If a change clearly made things worse, set the earlier rule again as a new version; nothing rolls back automatically, and one before/after comparison does not prove a policy works.
```

`POLICY-REVIEW` 是普通消息的约定前缀，不是控制消息：解析器只把 `role|policy|global` 加 `-policy|-proposal|-decision` 当作控制头，测试固定了这一点；relay 不解析、不拦截它，只供事后报告识别。

### 9.2 四种模式的角色块

#### off（静态 H）

master 与 worker 的新增披露均为**空字符串**，0 字符；不注入 B、不解析前缀/spawn 标记。既有文本原样保留，不加一句“当前关闭”，避免改变静态对照。

#### global（H-adaptive）：master 块 G-M

```text
Mode: global. The versioned communication policy C in the supplied snapshot applies to you and all workers. To replace it, call send_message with to="*" and message starting with GLOBAL-POLICY, optionally followed by base=v<k>. From the second line, provide the complete replacement C, at most 6000 characters. This control must be broadcast, not sent to just one worker.
Broadcasting attempts native notification to the current member list. It can include terminal members, have no recipients, or fail for some recipients; none of these outcomes undoes a committed update. New workers receive current C in their first request. Notification success is not proof that a worker has received a model request or followed C. Role-policy controls and formal proposal controls are not available in this mode.
```

#### global（H-adaptive）：worker 块 G-W

```text
Mode: global. Follow the versioned communication policy C in the supplied snapshot. The master can update C for the whole team. You cannot write formal policy state in this mode. Send ordinary suggestions to the leader when useful and continue your task under current C; a suggestion or update notification is not itself an approved policy change.
```

#### master（R-master）：master 块 R-M

```text
Mode: role policies. The supplied snapshot lists slots, current instances, versions, and full role policies. A role's approved communication policy overrides conflicting communication defaults in C or incidental assignment instructions, but not fixed runtime boundaries. Without a role policy, C applies.
To replace a slot's policy, send_message to that slot's current instance. Start the message with ROLE-POLICY <slot>, optionally followed by base=v<k>; from the second line provide the complete new policy, at most 2000 characters. Use the target slot's version in your supplied snapshot, not another slot's version. This control is single-recipient, never broadcast.
For a new slot, an agent spawn prompt may contain a line ROLE-POLICY: followed by its complete initial policy, ending at END-ROLE-POLICY or the prompt end. The text limit is also 2000 characters. For succession, instead include a line SUCCEEDS: <previous-name>. Use a new worker name, and only a predecessor the relay confirms as terminal. Do not combine the two markers. Succession keeps the slot and current policy version, not the previous conversation, workspace, or pending proposal.
Use ordinary messages for task coordination. A worker's ordinary suggestion does not change formal policy state; set the policy explicitly if you choose to adopt it. Global-policy updates are not available in this mode.
```

#### master（R-master）：worker 块 R-W

```text
Mode: role policies. For your communication choices, follow your approved role policy in the supplied snapshot. It overrides conflicting communication defaults in C or incidental assignment instructions, not fixed runtime boundaries. If no role policy exists, follow C. You cannot write formal policy state in this mode. Send ordinary suggestions to the leader when useful, and continue working under your current policy.
```

#### propose（R-propose）：master 附加块 P-M

```text
Formal worker proposals are enabled. Your supplied snapshot includes each pending proposal's ID, slot, proposer, base version, and full CHANGE/REASON/EVIDENCE. Judge that content, not a possibly stale notification. To decide, send_message to the proposer with first line POLICY-DECISION #<id> approve, POLICY-DECISION #<id> reject, or POLICY-DECISION #<id> amend. Approve/reject may have a NOTE: below the header, at most 2000 characters; amend must put the complete replacement policy below it, also at most 2000 characters, without another control header. Decisions are single-recipient, never broadcast. Approve when the four review questions hold; amend when the direction is right but the scope is too broad or the requirement is impractical; reject when evidence is missing or the proposal repeats an existing rule without a substantive change. Give the reason for an approval or rejection in NOTE, and for a rejection say what information is missing. The worker keeps working under its current policy while you decide.
A decision applies only to that pending ID. If it has been replaced or its base is stale, the whole response may be rejected. A reported replacement ID is diagnostic, not permission to transfer approval; read its full content in the next snapshot and decide again. Approval is a formal instruction, not a new messaging capability or proof of compliance.
```

#### propose（R-propose）：worker 块 P-W

```text
Mode: role policies with proposals. Follow your approved role policy from the supplied snapshot; it overrides conflicting communication defaults in C or incidental assignment instructions, not fixed runtime boundaries. If no role policy exists, follow C.
When concrete task evidence suggests a better rule for your own outgoing communication, you may send_message to="leader" with first line POLICY-PROPOSAL, optionally followed by base=v<k>. Below it provide CHANGE: with the complete replacement policy; REASON: the problem the current rule causes, why this change should fix it, and any added cost or risk; and EVIDENCE: the specific messages, tasks, or files, what you expect to observe afterwards, and at which related event it can be checked. These fields including their labels must total at most 2000 characters. The relay derives your slot and base from your request snapshot. Do not propose rules for other roles; request those through ordinary messages to the master instead. If your change depends on cooperation from another role, say so; proposing that another role confirm something is not a commitment from that role. Only the master can approve a proposal.
Continue working and communicating under the current approved policy while a proposal is pending; you do not need to wait for a decision. At most one proposal is pending for your slot. Resending identical CHANGE/REASON/EVIDENCE under the same base only retries notification and keeps its ID and submission time. Changing any of these fields replaces the old pending proposal with a new ID, even if the master previously received it. Repeated replacement can delay a decision. Read the proposal status in your next supplied snapshot; do not infer approval, rejection, or failure from native send results. The proposal itself does not change your active policy before approval.
```

### 9.3 D10 备选尾注（仅 propose worker；10-10 已定 D10-C）

两种尾注互斥，S0 实现前须按已确认的 D10 选择，并保证文本对应实际运行时；不能把“待决定”占位语句注入模型。未选不代表默认打开 compat 补丁；脚本测试可分别验证两种组合，真实模型短跑需先明确采用哪种。

#### D10-C：若采纳并验证 compat 判断

```text
Your end-of-turn text is forwarded to the leader automatically. Sending a formal proposal during the turn does not change that. A proposal is not a final task report; keep your task results clear and separate.
```

#### D10-D：若仅披露原生副作用

```text
Native reporting caveat: sending a formal proposal to the leader suppresses automatic forwarding of your final text for that turn. When you finish that turn's work, explicitly send the leader an ordinary task report. The proposal itself is not a task report. This requirement is additional to working under your current policy while a decision is pending.
```

### 9.4 组装与字符数

| 模式 | master 组装 | 新增字符数 | worker 组装 | 新增字符数 |
|---|---|---:|---|---:|
| off | 空 | 0 | 空 | 0 |
| global | B + Q + G-M + V | 4847 | B + Q + G-W | 3015 |
| master | B + Q + R-M + V | 5425 | B + Q + R-W | 3089 |
| propose | B + Q + R-M + P-M + V | 6806 | B + Q + P-W + D10-C / D10-D | 4728 / 4874 |

单块字符数（v3.4）：B 1455；Q 1209；V 1376；G-M 801；G-W 347；R-M 1379；R-W 421；P-M 1379；P-W 1849；D10-C 209；D10-D 355。以上按本文件代码块实际内容核计，每个块间连接另加 2 字符。v3.3 的对应总数为 global 2258 / 1804、master 2836 / 1878、propose 3798 / 3206。

本表只量化草稿新增披露，不把字符数换算为固定 token 数。动态块的策略、提案数量和全文长度不同，另按请求记录；使用 provider 实际 usage 计总成本。S0 要验证角色可见入口、长度、可选 base、广播收件人、stale 后纠错及 pending 不停工，再讨论遵从率；不得仅通过格式静态校验就宣称模型会按约定行动。

## 10. S1 实施记录（2026-10-10）

交付：`agent_company/policy/role_policy.py`（状态模块，含控制消息与 spawn 标记解析）与 `tests/test_role_policy.py`（44 项）。未接 relay、未改 `qwen.py`/compat、未调用模型；模式 off 不构造状态对象。

实现与 §3 的对应：暂存在副本上模拟整批，`commit()` 在最新状态上重新模拟、写一条 `role_policy_batch` 记录后才发布；写入或取消检查失败不留状态。四种前缀均以提交生效，模块里没有发送确认或回退路径。同文补发、异文替换、决定绑定确切编号、继任沿用槽与版本、终止作废待决提案、运行结束作废，均按 §3.1–3.2。重放与在线路径使用同一个变更应用函数；缺批、乱序、字段缺失或版本不连续时拒绝重建。

以下几处规格未写明，S1 做了选择，S2 前请确认：

| 项 | S1 的处理 | 理由 |
|---|---|---|
| 决定须引用 master 快照中出现过的提案 | 编号存在且待决、但不在本次请求快照里，返回 `stale_proposal` | 保证 master 只能批准它在注入里读到全文的提案 |
| 近似控制头 | 首行是连字符或下划线连接的关键词（含大小写变体、Markdown 修饰）但不合语法时，返回 `malformed_header`；带空格的普通英文如 "Role policy question" 仍按普通消息处理 | 避免写错的控制消息被当成普通消息静默送出 |
| spawn 标记近似行 | 与标记仅大小写不同的整行，或 80 字符以内形如 `succeeds: …` 的行，返回 `malformed_header` | 同上；长句不受影响 |
| 全局策略长度上限 | 运行中替换的 C 上限 6000 字符；配置的起始 C 不受限 | §3.2 未给数值，P1 起始 C 约 3.3k 字符 |
| NOTE 长度 | 上限 2000 字符 | 与角色策略一致 |
| 新增错误码 | `unknown_proposal`、`malformed_body`、`too_long`、`duplicate_tool_call`、`predecessor_not_terminal`、`stale_succession`、`conflicting_markers`、`invalid_name`、`run_ended` | §3.2 的错误码列表写的是"包括" |
| spawn 失败后的名字 | 失败的 planned 实例作废后，名字可再次使用；继任者失败则槽的当前实例恢复为前任 | 原生在 spawn 失败时会回滚成员表 |
| 槽数 | 不设固定槽数上限 | 10-13 取消 worker 人数上限后保持一致；单槽长度限制保留，S0 记录实际注入长度 |

给 S2 的约束：`commit()` 在持有状态锁时调用写入函数和取消检查，S2 不能在持有 relay `_lock` 时调用状态对象，否则与 `event()` 的加锁顺序相反，可能死锁。

### 10.1 D10 实测结果与适用范围（2026-10-10，S2c；10-10 审查后收窄）

审查意见和 v3.1 认为：worker 给 leader 发过提案后，原生就不再转发它回合末的文字汇报。真实 CLI 脚本场景只支持下面这个较窄的结论：

- **已观察到**：原生每次模型响应后触发回合文本事件，文本非空时把它设为待转发的最终汇报并清掉"已显式汇报"标记（`onRoundText`，Qwen 0.23.1）。在已测的场景里，worker 先给 leader 发消息（提案或普通消息都一样），之后的非空正文照常被转发。
- **已观察到**：在已测场景中，正文为空的结束响应（含只有推理内容的）被原生重新请求了一次。
- **未证明**：所有回合都以非空正文结束，或者原生在模型持续给出空响应时的处理方式。也没有在同一个提案场景下做补丁开、关的直接对照。

因此不能说"撤掉补丁完全没有损失"。D10-C 补丁保留，只在提案模式生效；暂不扩大，也不删除。D10-C 披露尾注原有一句"其他显式汇报仍可能压掉转发"，与已观察到的行为不符，已改为：Your end-of-turn text is forwarded to the leader automatically. Sending a formal proposal during the turn does not change that. A proposal is not a final task report; keep your task results clear and separate.（209 字符，原 280）。若要决定去留，应先补一个同一提案场景下补丁开、关的对照（需要一个仅测试用的开关）。

### 10.2 审查修正（2026-10-10，S2 提交后）

| 问题 | 修正 |
|---|---|
| spawn 失败时原生只把 `error.message` 交给模型，无 "Failed to spawn" 前缀，名字被永久占用 | 只认原生固定成功文本（`Teammate "<name>" is now running`），其余一律视为失败并作废 planned；发送结果同理；测试钉住原生成功文本；真实 CLI 场景：11 个 spawn 撞原生上限后同名重试不被拒 |
| 通知表只用 `tool_call_id` 作键，不同 worker 会互相覆盖 | 键改为（角色，调用编号），且只在答复的那次调用参数与已提交的一致时才计入，防止编号被复用时串号 |
| 注入事件写失败会留下在途标记 | 写入移入负责清理的块，失败按证据写入失败返回 424；预算拒绝分支同理 |
| attempt 记录缺 `source_request` | attempt 记录、重试事件、请求结束事件都带逻辑请求号 |
| 运行结束不作废从未启动的 planned 实例 | `end_run()` 记 `never_started` 并作废，槽恢复 |
| CLI 测试可能超时后仍通过 | 静默场景看到预期文本即下线；所有 CLI 场景断言 `native_cli_completed`、未超时 |

### 10.3 S3 实施记录（2026-10-10）

`programbench_evidence_projection.py` 升到 schema 4，原有字段不变，新增两段：

- **`prompt_cache`（所有运行）**：按角色统计 attempt 数、带缓存字段的 attempt 与其 prompt tokens、命中 tokens、命中占比。数据取自 attempt 记录里的原始用量，不用 relay 归一化后的数（归一化会把缺字段记成 0）。缺字段的 prompt tokens 单列，不进分母。
- **`role_policies`**：状态分三种，旧运行为 `not_instrumented`，记录为关闭的运行为 `off`，有账本的运行为 `on`。`on` 时包括：账本批次数与重放校验；每个版本的来源、正文与首次携带延迟（提交到第一次带上该版本、且已发起上游请求的 worker attempt）；每条提案的状态、补发次数、送达 master 的首次携带延迟、决定延迟；决定率与批准率（分母分别是提案数与已决定数）；各角色注入次数与长度；通知结果（无结果记 unknown）；拒绝的阶段、错误码、过期决定数与被连带丢弃的调用数；spawn 结果；广播的逐收件人结果；按版本统计的消息数与收件人分布；生命周期读取摘要。

协作时间线的 `send_message` 条目新增 `policy_control` 标注，策略控制消息与任务通信分开统计。每个条目新增 `role_policy_version`，即生成该调用的请求所携带的、调用者自身槽位的版本。attempt 时间线新增 `source_request`。

限制：按版本的消息分布只是描述，生效时长不同、任务阶段不同，不能据此推断因果。首次携带延迟用的是 relay 时钟下的 attempt 开始时间，不是模型读到的时间。

### 10.4 第二轮审查修正（2026-10-11）

| 问题 | 修正 |
|---|---|
| 同一角色复用调用编号时，历史里的旧结果会被当成新调用的结果 | relay 记下该响应在本角色历史中是第几条 assistant 消息，只认答复这条消息、且参数完整一致的结果；追踪键为（角色，逻辑请求号，调用编号）。compaction 改变位置时结果保持未知 |
| 晚拒绝后，投影读到的是没有返回给 CLI 的响应 | `_qwen_response.json` 改存实际返回的拒绝版本，暂存版另存为 `_policy_staged.json`；按版本的行为统计排除执行前被拒的调用 |
| 全局策略版本没有投影 | 版本表加入全局版本的正文与首次携带延迟；全局模式下行为按 C 版本归属；每次调用同时记录槽版本与全局版本 |
| 投影低估结果未知的通知 | 按（角色，调用编号，逻辑请求号）计算 |
| 注入写入失败被记为上游已启动 | 记为未启动；晚提交阶段的失败仍记为已启动；`provider_attempt_started` 也带逻辑请求号 |
| 低优先级：首批为空的损坏账本使投影报错；D10 旧措辞过强 | 投影改为记录重放失败；§0、§1.3、§5.2、§6 与测试说明按 §10.1 收窄 |

## 11. S0 准备（2026-10-11，未部署、未启动）

S0 跑两个一小时的 Team 运行：提案模式和全局模式。提案模式包含主控模式的全部控制消息，所以主控模式不单独跑。任务 FFmpeg，A/C 用已固定哈希的 P1，provider、资源和运行时开关沿用 10-03 运行，墙钟时间改为 3600 秒并加上 `role_policy_mode`。代码不是同一份：增加了角色策略机制。配置：`configs/stage1/programbench_ffmpeg_s0_{propose,global}_20261011.json`。

正式运行前依次做：部署快照并核对哈希、在 Linux 上跑角色策略测试、任务容器里的脚本冒烟（不调用模型）、有界真实冒烟（每个模式最多 40 次模型调用，单次尝试、120 秒请求期限，不覆盖正式运行的 11 次尝试与 1800 秒恢复路径；模型是否遵守约定只记录、不作为通过条件）、缓存探针（6 次调用：不变请求三次，更新一次策略状态后再三次，只记用量、缓存字段和延迟）。每一步都需要单独放行，不自动重试。启动器与步骤说明在本地 `local_docs/launch_programbench_ffmpeg_s0_20261011.py` 与 `runs/stage1_kimi_programbench_ffmpeg_s0_20261011/START.md`（按惯例不入库）。

S0 读的是协议使用情况、拒绝与纠正、注入长度、缓存字段与成本、运行健康，不读效应。模型完全不用约定也是结果，不据此重跑。

## 12. S0 修改依据、审批与复查提示（v3.4，2026-10-13）

状态更新（2026-10-14）：代码、测试和文档已随 `2f7bd45` 提交并 push。10-13 Kimi/FFmpeg v3.4 部署的两模式零 API 冒烟已通过（propose 10/10、global 9/9），Linux 测试 202 过、1 项旧人数限制测试失败；这些记录不覆盖后来取消人数上限与 DS 流式配置。10-14 已获准备性部署、测试、有界真实冒烟和缓存探针授权；S0、B、正式演化另确认。

**来源**：10-13 讨论认为，现有实现只保证"有记录地改策略"，没有规定什么问题值得改、按什么标准批准、改后看什么。决定先做最简单的一版：只改提示，不加独立 reviewer、不加自动门槛、不自动回滚，并入 S0 第一次运行。S0 本来就不是效应对比，所以无法、也不打算区分"有无这版提示"的差别。

**改了什么**（§9 是逐字版本）：

| 位置 | 内容 | 谁看到 |
|---|---|---|
| Q（新） | 什么时候值得改：有具体协作问题，或任务情况变了；考虑是规则不好、没遵守，还是消息还没送到；明显没遵守时先提醒；现有规则即使已覆盖，只要具体问题表明它实际不好用，仍可简化、澄清或改触发方式；原因未查明时写出假设，不要求先证明根因；看不到送达情况就如实说明；一次只改一个问题；证据要能定位（消息的发送者和时间、任务、commit、文件、check-in）；写明期望看到什么、在哪个相关事件后复查；不为复查而轮询；同一标准也适用于普通建议 | 三种模式的所有角色，紧跟 B |
| V（新） | 审批四问：有依据吗（原因未查明时是否写出假设）、对症吗（明显未遵守先提醒；没有实质变化的重复不批准；简化、澄清、改触发方式算实质变化）、能执行吗、值得吗；直接修改和 amend 的理由写在普通消息里，不混进策略正文；不要假定提交后立即影响了 worker，根据自然到来的汇报和任务事件复查，结论用普通消息、以 `POLICY-REVIEW <slot 或 C> v<k>:` 开头，写改善 / 无改善 / 副作用 / 暂无法判断加事实；无法确认行为是否受新版本影响时写暂无法判断，不为此轮询；变差就重设旧规则形成新版本，没有自动回滚，单次前后对比不证明有效 | 三种模式的 master，放在最后 |
| P-M | 何时批准、修订后批准、拒绝（没有实质变化的重复现有规则才拒绝）；批准和拒绝的理由写在 NOTE 里，拒绝要说明缺什么；等待决定期间 worker 照常工作 | propose 模式 master |
| P-W | REASON 写旧规则造成的问题、为什么能改善、额外成本或风险；EVIDENCE 写具体消息 / 任务 / 文件、期望看到什么、在哪个相关事件后复查；需要别人配合要写明依赖，"提议对方确认"不等于对方已承诺 | propose 模式 worker |
| launch.json | 新增 `role_policy_disclosure_version`（v3.4），off 模式为 null；已有 Git、部署清单和实际请求记录，不另加哈希 | 运行记录 |

原有的格式句与长度句一字未改，所以"每个执行的上限都已按执行值披露"的测试照常有效。

**没改什么**：控制消息格式、权限、长度上限、解析器、relay 的检查范围（仍不核验证据真假、不判断改后是否更好）、保存的 P1 A/C、S0 配置文件。`POLICY-REVIEW` 不是控制消息：解析器只识别 `role|policy|global` 加 `-policy|-proposal|-decision` 的开头，新增测试固定了"带这个前缀的消息按普通消息处理"。

**成本**：master 每次请求的固定披露增加约 2589–3008 字符，worker 增加约 1211–1522（§9.4 表）。S0 的注入长度和缓存探针都以 v3.4 为准，与 v3.3 的数字不可直接比较。

**测试**（复审修订后重跑）：角色策略、能力描述、S0 启动器、Team / Qwen / 投递 / check-in 相关测试共 330 项，329 项通过；唯一失败是 S0 启动器的快照检查，它要求部署路径没有未提交改动，属于预期行为，提交后应恢复。修订前的全量运行为 2921 项中 2919 项通过，另外 2 项（`test_script_import_edges_match_frozen_migration_allowlist`、`test_all_profiles_use_committable_policy_and_instruction_inputs`）在干净的 HEAD 上同样失败，与本次改动无关；修订后未重跑全量。这些测试只能证明文本拼装、限制披露和解析正确，不能证明模型会遵守审批与复查要求。

**复审修订（10-13）**：
1. V 原写"观察从相关 agent 的请求带上新版本开始"，但 master 的快照里没有这项记录，它判断不了。改为：不要假定提交后立即影响了 worker，根据自然到来的汇报和任务事件复查，无法确认就写暂无法判断，不为此轮询。精确的首次携带时间留给离线报告统计。
2. 原写"现有规则已要求就只能督促执行"过于绝对，会挡住合理修改，比如简化一条总被漏掉的复杂触发条件。Q、V、P-M 统一改为：明显未遵守先提醒；没有实质变化的重复不批准；允许依据具体问题简化、澄清或改触发方式；原因未查明时写出假设，不要求先证明根因。
3. 去掉 launch.json 里的披露哈希，只保留版本号。哈希只覆盖固定披露，不代表完整提示或实际送达，而 Git、部署清单和请求记录已经够用。
4. 状态措辞由"旧部署快照失效"改为"旧验证不覆盖 v3.4"。

**留到第二步（报告，尚未实现）**：

- 遵从率：提案四项是否写全、审批是否带 NOTE、直接修改后是否有说明、`POLICY-REVIEW` 的数量与结论分布。这些都是只靠提示的软要求，遵从与否要实测，不能假定。
- EVIDENCE 引用的存在性检查：引用的消息、任务、commit 是否确实存在。只查存在，不查真假。
- 把修改依据、预期和复查结论关联到策略版本；区分客观事件与 agent 自述，缺失就写未知；审批和复查消息本身的开销单独记录。
- 直接修改和 amend 的理由目前不进策略账本，报告里要标明"未强制记录"。

## 13. 取消 worker 数量上限（2026-10-13）

用户明确要求去掉 worker 数量上限。新版对所有 Team 模式（包括角色策略 off）取消 Qwen 原生的 10 人注册限制，不改变保存的 A/C，也不增加新的数量开关。兼容层通过原生 `Config.getAgentsSettings()` 为 Team 创建提供 `maxTeammates = Infinity`；不修改外部 Qwen 安装包。Single 不应用该覆盖，也没有获得 Team 工具。

master 的固定说明、`team_create` 工具说明和 check-in 均说明不设固定人数上限；状态文件、launch 和兼容层事件以 `unlimited` 表示，避免内部 Infinity 在 JSON 中变成 null。optimizer/reviewer 的目标能力说明同步此事实；历史投影只根据该次运行的记录判断，旧记录缺字段仍为未知。

只取消人数限制：共享 CPU、内存、墙钟、已配置的调用预算、工具执行并发、消息积压限制、循环保护、worker 权限和关闭流程保持不变。更多 worker 不增加资源或延长时限；不保证任意人数都能成功运行。

原“并发创建 11 人且固定 w10 应被拒绝”的测试不再适用于此设置，改为真实 CLI 配合脚本 provider 验证先 10 人、再第 11/12 人成功，独立工作目录、请求身份及全队正常收尾；非人数导致的 spawn 失败释放名字仍由现有 relay 测试覆盖。此次仅本地修改与零外部 API 验证，未更新已有 S0 部署；旧部署及其测试记录保留，不能当作本次改动的验证结果。

本地验证：12 worker 真实 CLI 脚本场景通过。476 项相关回归首轮为 471 通过、5 失败，失败来自尚未同步的 loop 测试夹具（缺少原生 Config getter）及写死提示段落顺序的断言；同步后该文件 18 项复跑全部通过，未放宽 loop 行为断言。其余回归覆盖 Single 权限、共享调用预算、超时/取消、消息投递、角色策略与资源配置；Ruff 和差异空白检查通过。未重新进行 Linux / 容器 cgroup 实测，未调用真实模型。

