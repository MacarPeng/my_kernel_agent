# RTX 6000 Ada: exact hardware boundary

Scope: NVIDIA RTX 6000 Ada Generation, compute capability 8.9 (SM89).
This is not RTX A6000 (Ampere), RTX 6000 Turing, or RTX PRO 6000 Blackwell.
The product has 48 GB GDDR6 ECC and advertised bandwidth 960 GB/s. These are
datasheet values, not measured sustainable bandwidth. Read actual SM count,
memory, UUID, driver and clocks on the rented machine; never copy B200's 148 SMs
or HBM figures. NCU percent-of-peak metrics already use the profiled device.

Per SM: 48 resident warps, 24 blocks, 65536 32-bit registers, 100 KiB shared
memory. Per block shared memory can reach 99 KiB with opt-in; static allocations
remain limited to 48 KiB. Actual residency also depends on block shape, register
allocation and shared-memory allocation. Occupancy alone does not prove a bottleneck.

Ada supports warp-level Tensor Core matrix operations, including FP8 hardware,
and the ordinary SM80 cp.async path. Compiler/dtype support must still be checked.
Software pipelining and software warp specialization can be valid. TMA bulk tensor
copies, WGMMA, thread-block clusters/DSM, Blackwell TMEM, tcgen05, hardware CLC and
native NVFP4 paths are outside this target. Use ordinary loads/stores, SM89-compatible
mma paths and CTA shared memory. Do not lower task precision/tolerances to gain speed.

Sources (vendor facts, checked 2026-10-30):
- https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
- https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/rtx-6000/proviz-print-rtx6000-datasheet-web-2504660.pdf
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

No local GPU measurements are asserted by this page.
