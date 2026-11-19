# GDN prefill: portable optimization hypotheses

Evidence class: portable hypotheses, not measured Ada speedups. Apply only after
reading the official task recurrence, dtype, sequence boundaries and state layout.

The small-token regime can be dominated by launch overhead and insufficient work;
larger inputs can expose state/matrix traffic, register pressure or compute limits.
Use measured Wi per-case latency and sequence/token groups to choose one question.
Do not assume the same bottleneck across all sequence lengths or distributions.

For a bandwidth question, test coalescing in k-last state layout, redundant state
loads/stores and fusion of gates/epilogues. Count bytes from the actual dataflow;
replay cache flushing can differ from normal benchmark behavior. For a register
question, inspect registers and achieved occupancy together, then try a smaller
tile, fewer live intermediates or a shorter pipeline. Increasing num_warps can
increase resources and regress latency; sweep narrowly around a supported hypothesis.
For a compute question, check whether legal SM89 Tensor Core shapes and the task's
precision contract allow the intended tl.dot. FP8 hardware is not permission to
quantize BF16/FP32 state or inputs in this task.

For ragged inputs, masked tail handling, batch scheduling and balanced CTA work can
be portable. Report launch grid and runtime SM count before attributing a tail.
PM sampling may help long kernels, but tiny captures need enough samples; it is
not required to propose a tail hypothesis. A source specialization must still pass
all 100 official cases; partial Wi improvements cannot promote the candidate.

Upstream provenance: KernelWiki gated-delta-net and register-pressure pages at
76d27b56f804e7e7295d4c570e1e5d7eef4b0a75, available in the frozen HAN assets.
https://github.com/mit-han-lab/KernelWiki/tree/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75
Those pages include SM90/SM100 experience. Their measured gains and TMA/WGMMA/TMEM
implementations are not SM89 evidence. The native SM89 query returns this curated
transfer discussion instead of relabeling historical kernels or emitting them as code.
