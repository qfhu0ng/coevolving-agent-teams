# Optimizer / reviewer 的 harness 能力输入与 A/C 可执行性方案

状态：修订 v2（2026-10-12，按审查意见四处修改）。7.1 两个小修正与 A–D 已在本地实现并通过零 API 测试，记录见第 11 节；未推送、未部署、未调用模型。§10 第 4、5 项仍待决定。

本方案基于本地 `abf4346` 的相关实现和现有测试代码进行只读核对。本次只新增本文档，没有修改代码、配置或 A/C，没有运行测试、连接服务器、调用模型、部署、提交或推送。工作区原有改动保持不动。

> 2026-10-14 更正：§1 撤回“单次噪声大于效应”的断言。Kimi 142 / 323 和 GLM 201 / 242 均跨运行条件变化，FFmpeg 尚无一组有分的同配置重复，噪声量级未知；启动记录与更正依据见《质量分析》B2（未公开的内部分析）。本注只修正文档依据，不改变本方案的能力检查范围或授权边界。

## 1. 结论与范围

**Optimizer 应在当前 harness 提供的动作、权限、观察和唤醒条件内修改 A/C；不能先假定一项能力存在，再由我们替它修改 harness。** Reviewer 则应独立核对候选策略是否有可实现的执行路径，而不只是评论策略可能有没有效果。

建议做四项有边界的改动：

1. 从候选实际使用的执行参数链生成精简的 `target_harness`，直接进入 optimizer 的初始输入；reviewer 复用同一份数据。
2. Optimizer 对新增或改变的策略机制提交简短 `execution_notes`：谁执行、何时能观察到触发条件、用什么现有动作、可接受什么延迟、不能及时执行时怎么办。
3. Reviewer 阅读完整 A/C，独立给出 `executability`，区分可支持、有条件支持、不支持和未知，不把 optimizer 的自述当证明。
4. 用普通单元测试、脚本 provider 和真实 CLI 的合成机制场景验证接线与运行语义；真实模型的遵从性和任务收益另行验证、另行授权。

**本方案只解决"策略能不能在目标 harness 上执行"，不解决"优化方向是否正确"。** 外循环暂停的主因是评分信号：官方总分被信息层主导，且缺少有分的同配置重复，运行条件变化与随机波动尚未分离，不能判定噪声量级或宣称其大于效应（`docs/stage1_framework_iteration_plan_20261005.md` T1、T2）。恢复正式演化前必须完成评分分层；若改用功能层作为目标，optimizer 的目标描述、reviewer 的判断依据和最终选择规则必须同时改为同一口径，官方总分仍保留并报告。可执行性检查做完而评分未分层时，不恢复正式演化。

首批接入 ProgramBench study / evolved followup 的 outer 路径。共享数据类型保持对旧调用方兼容；不顺手改变通用 evolution 的选题、采纳或评审协议。S0 没有 optimizer，本方案不把 optimizer 加进 S0，也不授权启动 S0。

不新增能力哈希、冻结合同、签名、百分比检查门或独立检索服务；不让 outer 模型读取服务器、凭据或完整源码；不增加新定时器；不人工改 P1 的 50% / 70% / 85%；不自动调用 optimizer 返修。

## 2. 现有实现核对

### 2.1 真正的缺口不是“完全没有 harness 说明”

| 当前实现 | 已经具备 | 缺口 |
| --- | --- | --- |
| `ProposalRequest` 与 optimizer 初始消息 | 当前 A/C、结果、历史、训练证据概览、selection 信息 | 没有候选将使用的 `target_harness`。见 C1、C2 |
| optimizer / reviewer system | 明确 runtime、工具、权限和资源固定；要求看机制、成本与证据 | 没有具体目标能力表，也没有逐条执行路径检查。见 C2、C3 |
| ProgramBench study 输入 | 说明 master 只协调、保持同一 runtime/check-in，训练只使用 baseline Team | 说明是概括性的，不提供完整有效参数和派生值。见 C4 |
| 历史证据 projection | 已有 `runtime_rules`、`submission_contract`、历史开关记录、角色策略与消息事件 | 这是源运行证据及当前投影器的解释，不是下一轮候选的目标能力。见 C5 |
| 证据阅读与引用 | 分页、JSON 指针、覆盖记录、file + quote 引用检查 | 概览不等于读过正文；引用存在不等于推论正确或适用于目标运行。见 C2、C6 |
| proposal / review 输出校验 | 格式、非空、长度、引用与 replay 选题检查 | 没有结构化可执行性说明；不会发现“策略依赖一个不存在的定时器”。见 C1–C3 |

`_proposal_request()` 目前只把 baseline Team 投影和任务摘要写进授权训练目录。Reviewer 的 `incumbent` 内嵌同一 `proposal_request.as_prompt_data()`，因此可以通过现有路径获得新字段，无须再开一个工具或扩大训练文件权限。

当前 projection 的 `RUNTIME_RULES` / `SUBMISSION_CONTRACT` 是当前投影器中的静态说明，部分 `effective` 叶子才来自源运行的 lifecycle 记录。对缺失、冲突或不适用的旧记录不能补成“已经确认开启”。

### 2.2 配置原文也不等于有效运行条件

现有执行链是：

`prepare_study / prepare_comparison → plan.inference → runner_inference_options → ProgramBenchRunner → QwenTeamRunner.run → native / relay`

以下默认值和派生行为已经存在，需要来自这条链，而不是另写一份常量表：

- 消息投递默认 `native_idle`；披露和 leader recovery 默认关闭；调用上限默认 `native_default`；角色策略默认 `off`。
- 这套 ProgramBench 入口要求显式 `max_model_calls=null`，并向 runner 传 `None`；不能推广成所有入口都无调用上限。
- Single 的有效角色策略是 `off`，Team 的 check-in、消息投递和 leader recovery 不能按 profile 原值宣称对 Single 生效。
- native 请求超时为 provider 请求期限加 `max(30 秒, 10% × 请求期限)`。
- Team 启用进展超时后，inactivity 会被提高到至少 native 请求超时加 60 秒，并与原配置/default 取较大值；这不延长任务总期限，也不复活 worker。
- coding relay 的逻辑期限跨尝试和退避共享；outer optimizer 请求的 `TimeoutError` 不重试。两套恢复语义不能混写。

### 2.3 “约 70% / 85%”应怎样理解

保存的 P1 A 写的是大约 50%、70%、85%，还包括 major package completion；它不是逐字承诺精确定时。当前 check-in 有周期观察以及默认 50%、75% 里程碑，但会合并、跳过或延迟。消息和 worker 回合末汇报也可能唤醒 master。

因此正确判断不是“70% / 85% 绝对做不到”，更不是“只能轮询才能做到”，而是：

- 当前没有保证在这些任意时间点唤醒 master 的专门调度能力。
- master 当时正在活动，或者随后收到消息/check-in 时，可以观察时间并执行一次近似检查。
- 如果策略要求精确时点、限定最大延迟，或隐含持续轮询且不承认其成本，就需要说明当前能力是否满足；不能以“大约”掩盖执行依赖。
- 即便 50%、75% 也不是“保证即时执行”的时刻，不能简单替换成两个数字就宣称问题解决。

本方案让 optimizer 自己选择可执行的触发与容差，不由实现者代它设计最优检查点。

## 3. 目标能力如何生成

### 3.1 使用同一参数解析结果，不维护两套配置

**运行语义的文字直接复用 agent 已经收到的说明，不另写一套。** 等待与唤醒、消息投递、loop 保护、check-in 时间表、角色策略控制规则，分别取自 `qwen.runtime_rules()`、`qwen.delivery_text()`、`qwen.checkin_system()` 与 `role_policy_runtime.disclosure()`，按本次实际配置调用（角色、check-in 间隔、投递模式、回合上限、角色策略模式）。这样 optimizer 读到的与 coding agent 读到的是同一段文字；一处修改，两边同时变化。只有 agent 提示里没有、而 outer 需要的事实（例如交付语义、评测时机、资源）才新写，且同样从现有常量或函数取值。

在准备候选 runner、已解析目标配置之后，且 optimizer 首次请求之前，生成 `target_harness`。优先复用 `runner_inference_options()`；对于只在 `QwenTeamRunner.run()` 内补齐的默认和派生值，抽出无副作用的解析函数，使 runner 与能力描述共同使用其结果。

该抽取是行为保持型重构：不改默认值、超时、权限或工具，不改变任何已有 safety check。用现有类型、版本记录和普通回归测试约束一致性，不再叠加一套“能力合同”系统。

来源分两类：

- **目标配置事实**：当前候选臂、已解析资源/时间/模式/开关及公开 provider 技术参数。
- **当前实现语义**：工具集合、等待条件、投递边界、check-in 合并规则、交付语义等，来自现有实现常量或同源描述函数；与相应机制测试配套。

能直接从常量和解析函数取的值，不在新模块手抄。跨 Python / JavaScript 的行为说明不能假装已经由一个数据对象完全证明，应由现有版本信息和跨层机制测试核对。无法识别 native 版本或无法确认某条语义时，写 `unknown`，不要沿用旧说明猜测。

目标代码使用现有 Git / 部署版本记录，原生依赖使用已有版本信息。不增加逐能力哈希。若已有记录无法说明某项来源，诚实保留未知，不能把一次代码审查写成执行实测。

### 3.2 输入内容与限制

| 类别 | `target_harness` 必须说明的内容 | 防止的具体误用 |
| --- | --- | --- |
| 适用范围 | Team / Single、角色策略模式、目标代码/原生版本的已知信息 | 把 Team 能力套给 Single，或把另一轮能力套给候选 |
| 角色与工具 | master 协调与只读工具；worker 编辑/shell/消息；spawn 支持的角色与工作区语义 | 让 master 写代码、让 worker 自行 spawn、假定异构模型选项可用 |
| 时间与预算 | 总任务期限、调用预算、有效请求期限、是否有进展恢复、何时到期 | 把重试误当额外任务时间，或安排截止之后的操作 |
| 等待与唤醒 | 周期、里程碑、合并/跳过条件；master 何时能进入原生等待，何时会结束运行 | 虚构任意时点 timer，或认为 master 随时能无限等待 |
| 可观察状态 | 状态刷新间隔；状态字段分别代表什么、缺什么 | 将分钟级快照当实时进展；反复读取却期待立即变化 |
| 消息 | 当前投递模式、生成/长工具期间不会被即时中断、发送/携带/执行的区别 | 把发送成功当成已读、理解、服从 |
| 在线策略 | 哪些模式支持改全局 C、改角色策略、自己提案；A 是否固定；版本何时进入请求 | off 模式要求 `POLICY-PROPOSAL`，或要求原地在线修改 A |
| 生命周期 | 失败、继任、slot 的含义和继承范围 | 把 `SUCCEEDS` 当成恢复旧对话或自动续上旧 worktree |
| 防护 | 实际生效的 cap、循环保护、受限恢复范围及终止后果 | 把关闭回合调用上限当成所有循环保护均已关闭 |
| 交付与评估 | 最终目录的文件交付、未提交文件也可导出、停止后独立评测、推理期不可读隐藏测试 | 以 commit=0 推断没有交付，或在策略里要求看官方隐藏失败 |
| 未知项 | 未记录、未验证或版本不匹配的字段 | 用缺失值补出能力保证 |

Provider 必须重新按叶子白名单提取。可包含 API 声明模型、输出上限、请求/进展期限、尝试次数和退避数值；不得复制 endpoint、环境变量键名、任意 `extra_body`、凭据、完整环境或启动命令。现有 `_public_provider()` 只移除 key，不能直接把它整个交给模型。

### 3.3 与历史证据严格分开

保留现有两个训练文件及读取边界。`target_harness` 是本次目标执行上下文，直接放在请求中，不伪装成 baseline 行为证据，不为它开放历史 `run_settings.json` 或服务器目录。

展示时清楚区分：

- **source observation**：历史 run 实际记录了什么，完整性如何。
- **target configuration / implementation**：候选按当前配置和实现预计获得什么能力。
- **execution observation**：以后真实运行记录了哪些开关与事件。本次尚无这层新增证据。
- **policy hypothesis**：在这些条件下采用某条规则可能改善效果，是待验证假设。

例如 source 采用 `native_idle`，target 采用 `between_tool_calls` 时，历史延迟仍是 source 的观察，不能改写成 target 已经发生过的延迟；也不能因新模式存在就宣称即时送达。

## 4. Optimizer / reviewer 如何获取并保留这些信息

### 4.1 接入现有请求对象

建议 `ProposalRequest` 增加可选 `target_harness`，并在 `as_prompt_data()` 中放入精简但完整的必要正文。ProgramBench 新协议路径提供该字段；旧通用调用方缺字段时记为“未提供目标能力”，不默认它已经通过可执行性检查。

只有启用本协议的新请求才增加对应 system 说明、提交工具字段要求和输出校验。无目标能力的旧调用方保持原消息与工具要求，结果展示层将缺失报告标为 `not_recorded`；不能把新字段在所有旧路径上突然设为必填。

`ReviewRequest` 经现有 `incumbent` 直接复用这份数据，不重新从 profile 或历史投影生成第二份。其 `candidate` 当前只列 A/C 和 rationale，实施时还需显式加入 optimizer 的 `execution_notes`，否则新增附件不会自动进入 reviewer 输入。候选对应的实际 runner 同样消费前述有效参数解析结果。

现有 outer working memory 会保留最初的 system/user 消息，因此放在初始 user 数据中的能力说明可随轮换保留。只放在可选训练文件里不能保证模型读取，也不能保证阅读页在上下文轮换后继续存在。

能力正文计入原有输入/上下文预算，不新增模型调用。控制体积，只列行为边界和实际参数，不注入完整源码、冗余部署清单或历史日志。必要时摘要也必须保留限制和未知项，不能只留下能力名称。

### 4.2 复用现有证据与保存路径

目标能力字段引用统一相对于共同的 `target_harness` 对象，例如 `/timing/checkin`；不相对于整条消息 JSON，因为 reviewer 输入中该对象位于 `incumbent` 内。历史观察继续使用现有 file + quote 引用。两类引用分别校验；不为迁就历史 citation checker 而扩张 TrainingEvidence 目录。

复用现有 request / proposal / review 保存记录，不新增独立的冻结文件或散落的副本。保存/恢复时保留新字段；旧记录缺字段显示 `not_recorded`，不得事后用今天的配置回填成当时输入。

目标配置发生改变时，旧评审只适用于旧目标。现有恢复路径不能拿更新后的运行配置默默套用旧评审；沿用现有新运行/输出目录和显式授权流程重新组织请求。此处不引入自动再评审、自动试跑或配置哈希门。

## 5. Optimizer 应提交什么

### 5.1 提示要求

在现有“runtime 固定”的说明后补充：

> 修改 A/C 时，以本次 target_harness 为执行边界；历史轨迹用于提出改善假设，不提供目标能力保证。对每个新增或改变的机制，说明执行者、可观察触发、现有动作、时间容差/退路和成本。目标能力未提供的动作不得写成已经可执行。需要修改 runtime 的想法单列为依赖未满足的建议，不混入声称可直接运行的策略。未能改善时允许保持 A/C 不变。

不是要求每句话填写表格，而是覆盖真正改变行为的机制，以及完整候选中仍保留的已知能力冲突。无须输出内部推理过程，只给可核对的简短执行说明。

### 5.2 `execution_notes` 结构

在 `PolicyProposal` / `submit_policy` 中增加附件，完整 A/C 仍按现有字段返回。建议每项包含：

| 字段 | 含义 |
| --- | --- |
| `policy_part` / `rule_quote` | A 或 C，以及候选原文中的短规则引用 |
| `actor` | 实际执行这条规则的角色 |
| `observable_trigger` | agent 如何知道条件成立；依据何种状态或收到的事件 |
| `existing_action` | 使用哪些当前允许的工具、等待或通信动作 |
| `capability_refs` | 支持该解释的 target_harness 字段路径 |
| `timing_and_fallback` | 允许延后多久或到哪个下一事件；无法满足时怎么做 |
| `cost_and_limitations` | 额外查询、消息、等待成本及尚未知的执行条件 |

这些字段是公开的设计说明，不是执行证明。普通校验只检查类型、引用存在、规则引文确实在候选中等可确定事项；不靠关键词或百分比正则决定策略可行性。

本协议的新请求必须包含结构合法的 `execution_notes`。无新增或改变机制、也没有已知保留冲突时，可以显式提交空列表并在 rationale 说明保持原样；不能靠遗漏字段绕过检查。旧 proposal 缺字段仍记 `not_recorded`，不追溯补写。

**真正影响执行的容差与退路必须写进 A/C 正文。** `execution_notes` 只给 outer 审查，不能依靠一个从未传给 coding agent 的附件来修正含糊策略。

例如 optimizer 可以提出“超过某阈值后，在下一次自然活动/唤醒时检查；没有新证据则结束回合等待”。这只是可行路径示例，不是替它规定最优检查点；若任务要求严格延迟上界，仍需证明当前事件来源能保证该上界。

## 6. Reviewer 如何检查可执行性

### 6.1 独立审完整 A/C

Reviewer 应先读目标能力，再读完整候选和 baseline 差异；不能只核对 optimizer 自己列出的 `execution_notes`。必须主动寻找未列出的前提，例如“精确唤醒”“worker 可自行创建继任者”“所有消息即时读到”“零 commit 即零交付”。

逐项核对：

1. actor 是否有这个动作的权限。
2. trigger 是否可被该 actor 观察，观察是在请求前、工具之间还是仅空闲时。
3. 时间要求是否由现有事件支持；条件性等待是否被误写成有保证的定时。
4. fallback 是否真实可用，是否会造成忙等、无限重发或提前结束运行。
5. 所需开关和角色策略模式是否有效；继承范围是否被扩大。
6. 成本与交付依据是否匹配其指标语义。
7. 历史证据是否适用于目标条件；缺记录是否被误当“没有发生”。

### 6.2 与质量推荐分开的 `executability`

`PolicyReview` / `submit_review` 增加独立区段，每项记录规则引用、相关能力字段、判定、简短原因与尚需验证的条件。

本协议的新 review 必须显式给出结构合法的 `executability`。审查后未发现问题与“没有做检查”要可区分，不能把缺字段当全通过；但 `unsupported` / `unknown` 是合法审查结论，不等于结构无效。结构完整性沿用现有提交校验和既有预算，不新增一轮返修调用。旧 review 缺字段记 `not_recorded`。

| 判定 | 含义 |
| --- | --- |
| `supported` | 根据已知实现与配置存在执行路径；不保证模型会遵从或效果更好 |
| `supported_with_conditions` | 只有接受某种延迟、观察条件或成本才成立，条件必须在 A/C 中说明 |
| `unsupported` | 依赖已知不存在或当前模式禁用的能力；说明具体缺口 |
| `unknown` | 当前输入无法确认，不用猜测补成支持或不支持 |

Reviewer 认为引文真实，并不等于策略正确；它认为可执行，也不等于机制已在新代码中跑通。两者分别由证据审查与脚本机制测试补充。

### 6.3 不悄悄改变已有评审权限

当前存在不同协议，不能用一个新 verdict 覆盖：

| 路径 | 当前语义 | 本方案默认处理 |
| --- | --- | --- |
| FFmpeg study / followup | 完整有效 review 是新 rollout 的前置；完整的负面/不确定推荐仍允许运行 | 保留；新报告明确展示可执行性问题，不把负面判定变成自动 veto |
| 通用 `review_replay`，固定 `first_n` | 负面或失败 review 不直接取消官方 replay，选择依据官方结果 | 保留，不借本次改动修改采纳规则 |
| 通用 `replay_selection=reviewer` | 缺合法选题时不能擅自回退；因此可能无 replay / inconclusive | 保留，这与 reviewer 对策略的否决不同 |
| `legacy_no_replay` | 无 reviewer，成功 proposal 可直接采纳 | 首批不宣称该路径获得了双重可执行性检查 |

`unsupported / unknown` 要在 dispatch 前写入 review / progress 摘要，并保留到最终报告，不能被一句“review completed”掩盖。**当前 followup 在完整有效 review 后直接 dispatch，没有中途人工确认点。** 保留现有协议意味着报告可供查阅，但不保证用户先读到结论再启动，也不保证已指出问题的候选不被运行。默认不新增自动阻断、人工等待、自动改写策略或第二轮 optimizer 调用。

如果希望机器自动阻止 `unsupported` 候选运行，或在 review 后等待用户确认，应单独确认这是调度/实验协议变更，并明确语义判断可能误报、由谁复核、是否额外调用模型及预算。它不能作为本方案的隐含附赠功能。

**这个决定不能无限搁置：下一次正式运行 optimizer 之前必须定下来**——是保持"完整评审后直接派发"，还是在评审报告出现 `unsupported` 或 `unknown` 时先等人工确认。届时把选择写进该次运行的启动记录。

## 7. 用具体问题检查方案是否有效

| 候选规则或判断 | 应如何评审 |
| --- | --- |
| “约 70% 时检查，在之后第一次自然唤醒时执行，允许延迟” | 查可用事件与等待条件；可能有条件支持，不能保证延迟上界 |
| “42 分钟时一定唤醒 master，即使没有消息或其他活动” | 没有对应专门 timer 时不支持；不得以轮询伪装成零成本调度 |
| “worker 收到更新后立刻中断当前长编译” | `between_tool_calls` 不保证中断正在运行的工具，不支持这一保证 |
| “commit 一直为 0，说明之前没有交付文件” | 证据推论不成立；交付是最终目录文件，Git history 不等于文件交付 |
| “off 模式的 worker 提交正式 POLICY-PROPOSAL” | 模式不支持；普通建议消息不等于正式策略状态修改 |
| “继任者自动获得原 worker 完整上下文和原 worktree” | 超出 slot / 策略继承语义；不能当成已有恢复能力 |
| “关闭 tool call cap 后所有 loop 均可自动恢复” | 错；其他循环保护仍存在，leader recovery 只覆盖受限场景 |

交付问题要分两层处理。当前 projection 已明确说明未提交文件可以交付，但检查点的 commit 指标仍不是文件变化指标。Runner 已有最终 `submission.files` / `source_bytes`，现有 task projection 未输出这些字段。可在后续小范围证据修正中投影这两个最终聚合值并声明“只说明最终导出，不提供整合发生时间”；不得据此编造历史文件变化曲线。本方案不顺手新增文件监视器。

另发现角色策略披露的一处覆盖缺口：解析器限制 decision `NOTE` 为 2000 字符，但对应披露没有明确写出这个 NOTE 上限；现有长度测试用文字断言，不能证明所有约束都与执行常量同源。实施时应补为一个独立、可审查的小修正。它说明不能把“已披露全部规则”当现成前提，本次未修改该代码或文本。

### 7.1 先行的两个小修正（v2 起先于 A–D 实施）

1. **NOTE 上限披露。** 在提案模式 master 的披露块（§9 P-M，`role_policy_runtime.DISCLOSURE_P_M`）写明 NOTE 的上限；披露长度测试改为从 `role_policy` 的执行常量取值，覆盖合法边界、超限拒绝和拒绝反馈里给出的上限。角色策略方案 §9.4 的字符数随之更新。
2. **最终导出指标进投影。** `task_projection` 输出 runner 已记录的 `submission.files` 与 `source_bytes`，并注明"只描述最终导出，不能据此判断过程中何时完成整合"。不新增文件监视器，不推断历史曲线。

两项都改变 S0 已部署的代码，S0 改用新的部署目录重新部署并重做零调用验证（Linux 测试、两个模式的脚本冒烟）；10-11 的旧部署目录和本地记录保留，不覆盖。

## 8. 分步实施与文件范围

各步先本地、零外部模型调用，完成后供审核；不自动提交或进入服务器部署。

| 步骤 | 主要文件/接入点 | 完成条件 |
| --- | --- | --- |
| A：有效能力解析 | `programbench_comparison.py`、`programbench_runner.py`、`qwen.py`，必要时小型 `harness_capabilities.py` | 同一有效参数用于 runner 与描述；默认/派生/角色适用范围不改变；未知信息显式保留 |
| B：outer 输入 | `evolution.py`、`programbench_study.py`、triad / followup 入口 | optimizer 与 reviewer 获得同一目标能力；历史训练文件范围不变；旧请求可读取 |
| C：输出与审查 | `optimizer.py`、`reviewer.py`、proposal/review 保存与 recover 路径 | 执行说明与独立评审区段可保存恢复；质量推荐和既有控制流程不变 |
| D：机制验证 | 对应 unit / runner / outer / CLI 合成测试 | 覆盖第 9 节，不把脚本回答当成真实模型判断力 |
| 先行小项（见 7.1） | `programbench_evidence_projection.py`、`role_policy_runtime.py` 及测试、角色策略方案 §9 | 已先于 A–D 实施；不扩大成历史状态采集工程 |

新 outer 输入和输出会改变后续演化的条件，应在已有版本/运行记录中注明启用，不声称与旧演化输入等同。它不会追溯改变已保存的 P1，也不会让旧结果自动获得新报告。

## 9. 零 API 验证计划与解读边界

| 测试 | 要验证什么 | 不能据此声称什么 |
| --- | --- | --- |
| 参数解析与 runner 接线 | 省略配置、显式配置、Single/Team、四种 role mode 的有效值一致；超时派生一致 | 所有依赖版本下都具有相同行为 |
| source / target 分离 | 历史投递/模式与目标不同时，两者不互相覆盖；未知保持未知 | 新模式已经改善历史问题 |
| 公开信息白名单 | 在配置放入秘密/endpoint/任意额外字段的哨兵值，outer 输入不出现它们 | 只删除一个 api_key 就已安全 |
| 上下文保留 | 初始输入、working-memory 轮换、submission-only 阶段均保留必要目标能力 | 模型一定阅读、理解和遵从 |
| 结构与保存恢复 | rule quote / capability ref、reviewer 的 candidate 附件、新协议必填区段、旧路径兼容、目标变更后的恢复范围 | 结构合法就等于语义可执行 |
| outer 脚本 provider | 脚本提交合法/错误结构、条件性/不支持示例，确认字段和报告贯通 | 脚本指定的 reviewer 标签证明真实模型能正确判断 |
| check-in / 等待机制 | 用合成时间和事件验证里程碑、周期、合并/跳过；必要时真实 CLI 脚本场景验证等待/唤醒 | 在每个名义时点保证即时模型执行 |
| 消息/角色机制 | 注入生效点、投递边界、权限与模式和能力说明一致 | observed 等于理解或行为执行 |
| 协议回归 | FFmpeg 完整负面 review 仍 advisory；失败 review 边界、通用 replay/legacy 语义不被误改 | 新增了自动 reviewer veto |
| 交付与披露小项 | 最终文件指标带适用范围；长度说明引用执行常量，拒绝路径有对应说明 | 可以从最终文件数恢复整合时间线 |

尽量使用现有测试设施，不再建立一套相似的 preflight 框架。凡是主张某条运行语义可靠，必须对应实际函数执行或 CLI 合成场景；仅检查配置中存在字段不够。

零 API 阶段可确认：目标事实接线正确、报告能保存、被披露的机制与被测运行路径一致。不能确认：真实模型会正确使用能力输入、会遵守 A/C、策略成本会下降、任务质量会提高。后者需要独立授权的真实模型验证；正式对照须保持模型、任务、harness、预算和采样安排可比。

## 10. 实施前需确认的决定

v2 已确定：

1. 先做 7.1 的两个小修正并重新验证 S0，再做 A–D；范围只覆盖 ProgramBench outer，通用入口保持兼容，不扩大评审权限。
2. 能力说明复用 agent 已收到的提示文本（§3.1）。
3. `executability` 与质量推荐分开，先作为必须展示的审查结果，不新增自动 veto 或返修循环。

仍待决定（下一次正式运行 optimizer 之前）：

4. 评审报告出现 `unsupported` / `unknown` 时，是否先等人工确认再派发（§6.3）。
5. 评分分层与选择口径（§1）：完成之前不恢复正式演化。

保留不做：不手改 P1 检查点、不新增定时器、不把"约 70%"直接判为不可执行、不加入轮询治理、压缩或 worker 自动恢复。

实施情况见第 11 节。未对现有 P1/S0 配置或策略作任何修改。

## 11. 实施记录（2026-10-12）

全部为本地改动和零 API 测试，没有调用模型、连接服务器、部署或推送。

**7.1 两个小修正**（已提交）：NOTE 上限写进 P-M 披露，所有 `*_MAX_CHARS` 由测试绑定到披露文字与解析器（`ef64c51`）；`task_projection` 输出 `final_export`（文件数、字节数，`scope: final_export_only`），未记录时为 null（`f59c923`）。S0 改用 `_20261012` 新部署目录，零调用验证通过（Linux 141 passed、两个模式脚本冒烟），10-11 旧部署与记录保留。

**A：有效能力解析。** 新增 `harness_capabilities.py`：`target_harness(inference, execution_mode, provider, code_version, native_version)` 从 `runner_inference_options()` 取有效值；`qwen.py` 抽出 `native_request_timeout_seconds()`、`team_inactivity_timeout_seconds()`、`CHECKIN_MILESTONES`，runner 与描述共用（行为保持，原有 qwen/team/leader-loop 测试通过）。规则、投递、check-in、角色策略文字分别由 `runtime_rules()`、`delivery_text()`、`checkin_system()`、`disclosure()` 按本次配置生成；agent 实际看不到的文字标 `shown_to_agents: false`。Provider 只取 `PROVIDER_FIELDS` 白名单。

**B：outer 输入。** `ProposalRequest.target_harness`（默认 None）；只有提供时 `as_prompt_data()` 才加入 `target_harness` 与协议说明，`ReviewRequest` 加入候选的 `execution_notes`（缺失记 `not_recorded`）和独立审查说明。只有 followup 与 triad 两个 ProgramBench 入口启用；通用 `run_stage1` 不变。

**C：输出与审查。** 启用协议时：optimizer 的 `submit_policy` 必填 `execution_notes`，reviewer 的 `submit_review` 必填 `executability`（至少一项）。校验只看结构：字段、取值、规则原文（空白归一后逐字出现在候选对应部分，8–600 字符）、JSON pointer 能在 `target_harness` 中解析；不按关键词判断可行性。不合格作为工具错误返回，在原有调用预算内改正。保存于原 proposal.json / review.json，旧记录缺字段恢复为 None（`not_recorded`）。Followup/triad 在派发前把 `executability_summary()`（各判定计数、`unsupported`/`unknown` 规则摘录、"dispatch unchanged"）写入 progress 的 `evolution.review_executability`；派发逻辑未改。

**D：测试。** `tests/test_harness_capabilities.py`（14）：S0 propose 配置下文字与源函数一致、哨兵秘密不出现；四种模式；省略配置时的默认值；Single 无 Team 区段；超时与实际启动的 `launch.json` 一致（3 种 provider 设置）；JS 调度器默认里程碑与 `CHECKIN_MILESTONES` 一致；RFC 6901；版本读取；摘要。`tests/test_executability_protocol.py`（42）：无协议时提示、工具与记录不变；协议下必填、保存、恢复；17 种不合格 notes 与 15 种不合格 executability 均先反馈再改正；凭据脱敏；旧记录恢复；working-memory 轮换与最后只能提交的一步仍保留 `target_harness` 和必填要求。Followup/triad 各加一个入口测试（`unsupported`/`unknown` 在派发前可见，且不阻止派发），followup 的投递模式测试加了"描述与 runner 同源"断言。顺带修正 `test_role_policy.py` 并发测试的偶发失败：`export()` 的 proposals 用字符串键，pending 用整数，结束时恰有待决提案才会比较失败；只改测试比较，不改实现。

**未做或保留未知：**

- `code_version`：入口没有现成的进程内版本记录可读（不在入口里调用 git），记为 unknown；部署目录的 DEPLOYMENT.json 仍是版本依据。`native_version` 读 Qwen 包的 package.json，读不到时为 unknown。`teammate_limit` 为 unknown（native 常量，未在本层核实）。
- 目标变更后的恢复：`recover()` 不比较目标配置；followup/triad 拒绝复用已有输出目录，所以这两个入口不会把旧评审套到新目标上。按 §4.2 未加配置哈希门。
- 测试只证明接线、保存和结构校验；脚本给出的 notes 与 verdict 不代表真实模型的判断力。
- §10 第 4 项（`unsupported`/`unknown` 是否先等人工确认）与第 5 项（评分分层）仍待决定；在此之前不运行正式 optimizer。

## 附录：本次核对的代码入口

以下是本地代码定位，不是本次新功能已经通过测试的证据。

| 编号 | 来源 |
| --- | --- |
| C1 | [evolution.py：ProposalRequest / ReviewRequest / 输出结构](../agent_company/evolution/loop.py)；[现有 review/replay 路径](../agent_company/evolution/loop.py) |
| C2 | [optimizer.py：工具与 system](../agent_company/evolution/optimizer.py)；[初始消息](../agent_company/evolution/optimizer.py)；[提交校验](../agent_company/evolution/optimizer.py) |
| C3 | [reviewer.py：system、输出校验与输入](../agent_company/evolution/reviewer.py) |
| C4 | [programbench_study.py：Study 输入说明](../agent_company/programbench/study.py)；[训练证据构造](../agent_company/programbench/study.py)；[followup 评审后的控制流](../scripts/run_programbench_evolved_followup.py) |
| C5 | [programbench_evidence_projection.py：交付与运行说明](../agent_company/analysis/evidence_projection.py)；[历史 effective 投影](../agent_company/analysis/evidence_projection.py) |
| C6 | [evidence.py：证据索引](../agent_company/evolution/evidence.py)；[outer_budget.py：initial 保留](../agent_company/evolution/outer_budget.py) |
| C7 | [runner_inference_options](../agent_company/programbench/comparison.py)；[ProgramBenchRunner 有效参数与执行](../agent_company/programbench/runner.py)；[qwen.py 超时派生](../agent_company/runtime/qwen.py) |
| C8 | [qwen.py：消息与等待披露](../agent_company/runtime/qwen.py)；[check-in 实现](../agent_company/runtime/qwen_team_checkin.mjs)；[relay 角色工具集合](../agent_company/runtime/relay.py) |
| C9 | [角色策略披露](../agent_company/policy/role_policy_runtime.py)；[策略长度常量](../agent_company/policy/role_policy.py)；[NOTE 校验](../agent_company/policy/role_policy.py) |
| C10 | [P1 检查点原文](../configs/stage1/policies/kimi_p1_20261001/assignment_v001.md)；[最终文件导出指标](../agent_company/programbench/runner.py)；[source boundary](../agent_company/programbench/workspace.py) |
| C11 | [现有 public provider 处理](../agent_company/roadmap/cli.py)；[followup 训练范围测试](../tests/test_programbench_evolved_followup.py)；[advisory 推荐测试](../tests/test_programbench_evolved_followup.py) |
