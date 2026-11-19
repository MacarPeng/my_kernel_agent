# Targeted NCU diagnosis on Ada

Evidence class: tool semantics and diagnostic hypotheses; installed tool/report wins.
Check ncu --version and --list-sections before capture. Select one tested workload,
one exact JIT kernel and one recipe: triage, memory, occupancy, compute or scheduler.
There is no --set full, mandatory source-counter pass or automatic PM-sampling pass.
One matching launch can still require several NCU replay passes: measure capture wall
time and profiler__replayer_passes when available. NCU timings are diagnostic and
must not replace the uncontaminated paired benchmark latency.

Useful metric families: gpu__time_duration.*, sm__throughput.*, dram__throughput.*,
launch__registers_per_thread, sm__warps_active.*, smsp__warps_eligible.* and
smsp__warp_issue_stalled_* / smsp__pcsamp_warps_issue_stalled_*.
Suffixes vary by tool/device. Preserve the exact name, numeric value and unit;
do not apply the original B200 metric renaming table to all GPUs. Missing/None/NaN
is unavailable, never zero. A recipe missing a required family is partial evidence.
High traffic or low occupancy alone does not establish a causal bottleneck.

PM sampling is supported on Ada with compatible NCU/platform permissions. It is
disabled in the default bounded recipes to control cost, not because SM89 lacks it.
vGPU and container/performance-counter permissions are separate limitations.
ERR_NVGPUCTRPERM requires the rental host's counter access; preserve the error and
continue with timing/compilation evidence. Do not repeatedly retry or escalate to full.

During pipeline execution, captures use the shared GPU lane only between Wi timing
blocks. CPU planning/wiki work can overlap evaluation; capture and kernel replay
consume GPU time and are extra work. Reports retain the frozen candidate hash and
GPU UUID. Never attribute an old version's counters to current edited code.

Sources:
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
