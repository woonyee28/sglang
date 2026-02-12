Logical Flow:
1. Baseline BF16 Grouped GEMM?✅
2. FP8 Group GEMM implemented by myself✅
3. FP8 2:4 structured expert weight pruning, leveraged NVIDIA hardware. 
4. Mixed precision FP4 weights and FP8 activations
5. all nvfp4 weights and activations✅

metrics:
1. latency
2. tflops / watt (essentially throughput and energy efficiency)
3. cosine similarity towards the baseline, shows the numerical fidelity of the GEMM itself

experiment desgin:
use deepseek R1
- different batch size
- different sequence length
- different number of experts, etc???

for my benchmark,
- i am thinking to focus on 1,2,5. as well as fully understanding the performance gap at kernel level, including all the scheduling algo, memory access pattern, etc
- add more different shape of matrix to simulate hardware architecture and how to design efficient moe architecture etc (based on different TP and EP parallelism too!)

## What the Graph Should probably Look Like
```
TFLOPS
  │                           
  │                         ★ Stage 4 (nvfp4)✅
  │                     ★ Stage 3 (transpoe trick for small M)
  │                 ★ Stage 2 (tile shape tuning) ❌ many tile sizes not supported so cant tune much anyways
  │         ★ Stage 1 (blockwise FP8)✅
  │   ★ Stage 0 (FP16 baseline)✅
  │
  └──────────────────────────────────────── Matrix Size