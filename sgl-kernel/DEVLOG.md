Logical Flow:
1. Baseline BF16 Grouped GEMM?
2. FP8 Group GEMM implemented by myself
3. FP8 2:4 structured expert weight pruning, leveraged NVIDIA hardware. 
4. Mixed precision FP4 weights and FP8 activations
5. all nvfp4 weights and activations

metrics:
1. latency
2. tflops / watt (essentially throughput and energy efficiency)
3. cosine similarity towards the baseline, shows the numerical fidelity of the GEMM itself

experiment desgin:
use deepseek R1
- different batch size
- different sequence length
- different number of experts, etc???