# Separate SM86 scope

The retained compatibility profile is NVIDIA RTX 3090, compute capability 8.6.
SM89 product capacities, FP8 support and register/shared-memory tuning decisions
must not be transferred to this profile as facts. Read runtime device attributes
and the NVIDIA Ampere tuning guide. This page prevents an Ada query fallback from
quietly assigning Ada limits to a 3090.

Source: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
No measured speedup or benchmark result is supplied.
