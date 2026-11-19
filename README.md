# KernelLab V1 单卡异步优化系统

基于 [设计规格](./design_20261118/IMPLEMENTATION_PLAN.md) 与 [整体架构](./design_20261118/ARCHITECTURE.png) 实现的 KernelLab V1 单卡异步优化运行时。

## 架构与模块映射

| 模块 | 对应角色 / 职责 |
|---|---|
| `kernel_agent_v1/protocol.py` | 统一调度消息头（禁止正文进入 Controller）、指令、不可变代码包清单、任务、版本与预算结构 |
| `kernel_agent_v1/store.py` | 追加写事件日志（`events.jsonl`）、原子状态快照、调度决策解释日志与证据落盘 |
| `kernel_agent_v1/duration.py` | 分组耗时统计（中位数典型估计、P90 偏慢敏感性检查、超时下界截尾处理） |
| `kernel_agent_v1/wiki.py` | `KernelWiki`：硬件知识插件与实验路线树（3 轮无明显进步自动暂停、失败/修复独立计数） |
| `kernel_agent_v1/leader.py` | `Leader` 与 `Researcher`：强制每轮首查 Wiki 凭据、假设与等待声明、跨路线优先级管理 |
| `kernel_agent_v1/coder.py` | `Coder`：指令约束检查（越界改动与凭据校验）、SHA-256 不可变代码包封存与完整性校验 |
| `kernel_agent_v1/evaluator.py` | 可信 `gdn-unified-v2` 评测协议：100 固定 UUID、4×25 分块、几何平均加速比、ABBA 复测与三块终评 |
| `kernel_agent_v1/operator.py` | `Operator`：`start`/`wait`/`stop` 统一接口、单卡独占租约与清理确认、CPU/GPU 拆分、双通道投递 |
| `kernel_agent_v1/judge.py` | `Judge`：完整评测核验、条件 ABBA 触发与提交时 incumbent 校验、终评验证等级判定 |
| `kernel_agent_v1/controller.py` | `Controller`：唯一状态/租约/版本号/`VERSION_CLOSED` 写入方，按间隔/边界批量落盘快照与恢复 |
| `kernel_agent_v1/scheduler.py` | `DynamicScheduler`：先筛选再窗口 `H` 成员活跃秒数打分、P90 敏感性回退与解释日志 |
| `kernel_agent_v1/backends.py` | 可插拔后端：`SimulatedExecutionBackend`、`LocalSubprocessBackend`（含非阻塞异步等待）与 `RealSingleGpuBackend`（`CUDA_VISIBLE_DEVICES` / `nvidia-smi` / NCU CSV 解析） |
| `kernel_agent_v1/async_runtime.py` | `AsyncKernelLabRunner`：基于 `asyncio` 的真实并发事件循环，重叠 GPU 子进程、CPU 解析与模型研究/编码 |
| `kernel_agent_v1/llm.py` | DeepSeek LLM 接入接口（默认离线安全占位，含 `DeepSeekClient`、`ScriptedDeepSeekClient` 与 Researcher/Coder 适配器） |
| `kernel_agent_v1/runtime.py` | `KernelLabRuntime`：串联五角色与事件循环，统计首份可行动证据、成员/GPU 重叠与吞吐指标 |
| `kernel_agent_v1/closed_loop_tester.py` | `ClosedLoopTester`：无卡闭环验证器，支持纯模拟与真实子进程执行并自动核验 10 项闭环不变量 |
| `kernel_agent_v1/gdn_task.py` | 复用适配自 `KernelLabPanel_20261018` 的官方 GDN Prefill 100 固定 UUID、分层 `W1..W4` 分块与 ABI/AST 校验 |
| `kernel_agent_v1/gdn_gpu_runner.py` | 真实单卡（Colab L4 / Ada / Ampere / Hopper）GDN 评测、ABBA、终评与 `ncu --csv` 采集执行入口 |
| `kernel_agent_v1/experiments.py` | `whole_round` / `chunked_fcfs` / `chunked_dynamic` 同协议三路基线对照 |

## 运行验证

运行完整单元测试与第 9 节最小验收清单测试：

```bash
python3 -m unittest discover -s tests -v
```

运行无卡完整闭环测试器（支持虚拟时钟 `sim` 和真实子进程隔离执行 `subprocess`）：

```bash
python3 -m kernel_agent_v1.cli closed-loop --backend sim
python3 -m kernel_agent_v1.cli closed-loop --backend subprocess
```

运行三路调度基线对比报告：

```bash
python3 -m kernel_agent_v1.cli compare
python3 -m kernel_agent_v1.cli compare --json
```

在 Colab L4 上运行真实单卡 GDN 闭环（先安装冻结 FLA `v0.5.2` 基线，再用种子 kernel 冒烟或加 `--deepseek-live` 接入 DeepSeek）：

```bash
# 0) 安装冻结 FLA v0.5.2 测速基线（加 --no-deps 避免覆盖 Colab 自带的 torch/triton）
pip install --no-deps git+https://github.com/fla-org/flash-linear-attention.git@9c8e42e762fce087c27b673af4922795d9edb85e

# 1) 无卡模拟干跑（使用官方 100 GDN UUID + 分层 W1..W4）
python3 -m kernel_agent_v1.cli run --backend sim --gpu NVIDIA-L4

# 2) Colab L4 真卡冒烟（种子 kernel 跑通真实 CUDA Event 测速）
python3 -m kernel_agent_v1.cli run --backend real-gpu --gpu NVIDIA-L4 --run-dir /content/drive/MyDrive/kernellab_runs/smoke1 --smoke-cases 4

# 3) Colab L4 真卡 + DeepSeek 自主优化（需先设置 DEEPSEEK_API_KEY）
export DEEPSEEK_API_KEY="sk-..."
python3 -m kernel_agent_v1.cli run --backend real-gpu --gpu NVIDIA-L4 --run-dir /content/drive/MyDrive/kernellab_runs/run1 --deepseek-live --max-rounds 4
```

实卡运行时，普通任务默认超时为 1800 秒，单次诊断上限为 450 秒，并受诊断有效期约束。`--abba-retest-timeout-s` 默认 18000 秒，`--final-eval-timeout-s` 默认每个终评块 36000 秒。默认整轮墙钟和 GPU 调度预算会计入 ABBA 和三块终评的预留时间。手动指定的超时和整轮预算均按传入的秒数生效。


### V500 长实验

使用 `--deepseek-live --max-versions 500` 运行路线搜索，种子计作 V1。状态切分、短序列路径、分块递推、变长任务分配、状态布局和门控融合分别注册为子路线。三次完整评测无明显进步会暂停对应路线。候选编译、执行或数值失败时，同一失败链最多提交两份修复源码，修复耗尽后切换到下一条可运行路线。相同源码只接纳一次。

所有已知子路线暂停后，Design Researcher 必须根据现有实验回执提出新的支路。新支路须指定父路线、对应的回执、不同的优化机制和可证伪的测试。新支路继承同样的停止规则，并记录在 `run_routes.json` 和 `lifecycle.jsonl` 中。运行时校验父路线与回执的归属，按标准化文本拦截重复假设和机制。

达到版本上限、预算接近耗尽，或三次开支路请求都未得到合格提案时，搜索结束，并对保留下来的最优版本发起一次终评。F1、F2、F3 是该次终评内部的三次测量。`run_result.json` 记录实际完成的版本数、是否达到上限和停止原因。

`run_config.json` 记录启动配置。`run_routes.json` 记录父子路线、停滞计数、修复计数和暂停原因。`lifecycle.jsonl` 记录版本接纳、GPU 任务启动及终评排程。`events.jsonl` 和 `scheduler_decisions.jsonl` 记录事件和调度决定。`evidence/` 和 `packages/` 保存评测证据、判定、模型调用和候选源码。`pipeline_timeline.html` 汇总工作时间。指定版本数时自动开启模型调用日志，不记录 HTTP 鉴权头。
