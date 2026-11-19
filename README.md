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

在 Colab L4 上运行真实单卡 GDN 闭环（先用种子 kernel 冒烟，再加 `--deepseek-live` 接入 DeepSeek）：

```bash
# 1) 无卡模拟干跑（使用官方 100 GDN UUID + 分层 W1..W4）
python3 -m kernel_agent_v1.cli run --backend sim --gpu NVIDIA-L4

# 2) Colab L4 真卡冒烟（种子 kernel 跑通真实 CUDA Event 测速）
python3 -m kernel_agent_v1.cli run --backend real-gpu --gpu NVIDIA-L4 --run-dir /content/drive/MyDrive/kernellab_runs/smoke1 --smoke-cases 4

# 3) Colab L4 真卡 + DeepSeek 自主优化（需先设置 DEEPSEEK_API_KEY）
export DEEPSEEK_API_KEY="sk-..."
python3 -m kernel_agent_v1.cli run --backend real-gpu --gpu NVIDIA-L4 --run-dir /content/drive/MyDrive/kernellab_runs/run1 --deepseek-live --max-rounds 4
```
