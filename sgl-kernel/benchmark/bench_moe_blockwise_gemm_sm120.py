import argparse
import os
import threading
import time

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
from sgl_kernel import fp8_blockwise_scaled_grouped_mm  # fp8 group scaled matmul
from sgl_kernel import cutlass_fp4_group_mm, scaled_fp4_quant  # fp4 group scaled matmul

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# try to import pynvml for power measurement
try:
    import pynvml

    pynvml.nvmlInit()
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_NVML = True
except Exception:
    HAS_NVML = False
    print("WARNING: pynvml not available. Energy metrics will be skipped.")
    print("Install with: pip install nvidia-ml-py\n")

# Consistent alignment across all kernels for fair comparison
M_ALIGNMENT = 128
NK_ALIGNMENT = 128


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align_up(x: int, alignment: int) -> int:
    return ceil_div(x, alignment) * alignment


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (128 - (n % 128)) % 128
    x = torch.nn.functional.pad(x, (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128),
        dtype=x.dtype,
        device=x.device,
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
        x_view.size(0), x_view.size(2)
    )


class PowerSampler:
    """Sample GPU power draw via NVML.

    Note: nvmlDeviceGetPowerUsage updates roughly every 500ms, so we poll
    at a matching interval to avoid redundant reads of stale values.
    """

    def __init__(self, interval_ms: float = 500.0):
        self.interval_s = interval_ms / 1000.0
        self.samples = []
        self._running = False
        self._thread = None

    def start(self):
        if not HAS_NVML:
            return
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        if not HAS_NVML:
            return 0.0
        self._running = False
        self._thread.join()
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples) / 1000.0  # mW -> W

    def _sample_loop(self):
        while self._running:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(NVML_HANDLE)
                self.samples.append(power_mw)
            except Exception:
                pass
            time.sleep(self.interval_s)


def compute_bf16_reference(
    n: int,
    k: int,
    num_groups: int,
    a_bf16: torch.Tensor,
    b_bf16: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """Compute BF16 grouped GEMM as reference for accuracy comparison."""
    total_m = expert_offsets[-1].item()
    ref_out = torch.empty((total_m, n), device="cuda", dtype=torch.bfloat16)

    for g in range(num_groups):
        start = expert_offsets[g].item()
        end = expert_offsets[g + 1].item()
        ref_out[start:end] = a_bf16[start:end] @ b_bf16[g].t()

    return ref_out


def accuracy_metrics(test: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    """Compute multiple accuracy metrics between test output and BF16 reference."""
    t = test.float().flatten()
    r = ref.float().flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        t.unsqueeze(0), r.unsqueeze(0)
    ).item()
    abs_err = (t - r).abs()
    max_abs_err = abs_err.max().item()
    # Relative RMSE: RMSE / RMS(reference)
    rmse = abs_err.pow(2).mean().sqrt().item()
    ref_rms = r.pow(2).mean().sqrt().item()
    rel_rmse = rmse / ref_rms if ref_rms > 0 else float("inf")
    return {
        "cosine_similarity": cos_sim,
        "max_abs_error": max_abs_err,
        "relative_rmse": rel_rmse,
    }


def run_benchmark_loop(run_fn, num_warmup, num_run):
    for _ in range(num_warmup):
        run_fn()
    torch.cuda.synchronize()

    power_sampler = PowerSampler(interval_ms=500.0)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    power_sampler.start()
    start_event.record()
    for _ in range(num_run):
        run_fn()
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    avg_power_w = power_sampler.stop()

    avg_time_us = start_event.elapsed_time(end_event) / num_run * 1000  # us
    return avg_time_us, avg_power_w


def compute_metrics(avg_time_us, avg_power_w, total_m, n, k, acc_metrics):
    avg_time_s = avg_time_us / 1e6
    flops = 2 * total_m * n * k
    tflops = flops / avg_time_us * 1e-6

    if avg_power_w > 0:
        energy_mj = avg_power_w * avg_time_s * 1000
        tflops_per_watt = tflops / avg_power_w
    else:
        energy_mj = 0.0
        tflops_per_watt = 0.0

    return {
        "total_m": total_m,
        "n_aligned": n,
        "k_aligned": k,
        "latency_us": avg_time_us,
        "tflops": tflops,
        "avg_power_w": avg_power_w,
        "energy_mj": energy_mj,
        "tflops_per_watt": tflops_per_watt,
        **acc_metrics,
    }


# ---------------------------------------------------------------------------
# BF16 Grouped GEMM baseline (torch.bmm — single batched cuBLAS call)
# ---------------------------------------------------------------------------
def bench_bf16(
    expected_m_per_group: int,
    n: int,
    k: int,
    num_groups: int,
    num_warmup: int,
    num_run: int,
) -> Dict:
    """
    BF16 grouped GEMM baseline using torch.bmm.

    All experts use the same (padded) M so we can issue a single batched
    cuBLAS GEMM, which is far more representative of an optimised baseline
    than looping with torch.mm per expert.
    """
    device = "cuda"
    n_g = align_up(n, NK_ALIGNMENT)
    k_g = align_up(k, NK_ALIGNMENT)
    out_dtype = torch.bfloat16

    m_g = align_up(expected_m_per_group, M_ALIGNMENT)
    total_m = m_g * num_groups

    # Batched layout: [num_groups, m_g, k_g] and [num_groups, k_g, n_g]
    a_bf16 = torch.randn((num_groups, m_g, k_g), device=device, dtype=out_dtype)
    b_bf16 = torch.randn((num_groups, k_g, n_g), device=device, dtype=out_dtype)
    c_out = torch.empty((num_groups, m_g, n_g), device=device, dtype=out_dtype)

    def run_fn():
        torch.bmm(a_bf16, b_bf16, out=c_out)

    # Accuracy: BF16 vs BF16 = perfect
    acc = {
        "cosine_similarity": 1.0,
        "max_abs_error": 0.0,
        "relative_rmse": 0.0,
    }

    avg_time_us, avg_power_w = run_benchmark_loop(run_fn, num_warmup, num_run)

    return compute_metrics(avg_time_us, avg_power_w, total_m, n_g, k_g, acc)


# ---------------------------------------------------------------------------
# FP8 Grouped GEMM Benchmark
# ---------------------------------------------------------------------------
def bench_fp8(
    expected_m_per_group: int,
    n: int,
    k: int,
    num_groups: int,
    num_warmup: int,
    num_run: int,
) -> Dict:
    """
    FP8 grouped GEMM benchmark.

    IMPORTANT layout note: per_block_cast_to_fp8(b_bf16.t()) produces [k_g, n_g]
    FP8 data with scales indexed to match that layout. We store the transpose
    [n_g, k_g] into b_stack, then call b_stack.transpose(1,2) to create a
    *non-contiguous view* [num_groups, k_g, n_g]. The kernel reads through this
    strided view, which preserves the correspondence between weight bytes and
    their scale factors. Making this contiguous would shuffle the bytes without
    updating scales, producing garbage output.
    """
    device = "cuda"
    # Use same alignment as other benchmarks for fair comparison
    n_g = align_up(n, NK_ALIGNMENT)
    k_g = align_up(k, NK_ALIGNMENT)
    out_dtype = torch.bfloat16

    m_g = align_up(expected_m_per_group, M_ALIGNMENT)

    expert_offsets = torch.zeros((num_groups + 1), device=device, dtype=torch.int32)
    problem_sizes = torch.zeros((num_groups, 3), device=device, dtype=torch.int32)
    layout_sfa = torch.zeros((num_groups, 5), device=device, dtype=torch.int32)
    layout_sfb = torch.zeros((num_groups, 5), device=device, dtype=torch.int32)

    a_tensors = []
    b_tensors = []
    a_scales_tensors = []
    b_scales_tensors = []
    a_bf16_tensors = []
    b_bf16_tensors = []

    for g in range(num_groups):
        expert_offsets[g + 1] = expert_offsets[g] + m_g
        problem_sizes[g][:] = torch.tensor([m_g, n_g, k_g], device=device)

        a_bf16 = torch.randn((m_g, k_g), device=device)
        b_bf16 = torch.randn((n_g, k_g), device=device)

        a_bf16_tensors.append(a_bf16.clone())
        b_bf16_tensors.append(b_bf16.clone())

        a_fp8, a_scale = per_token_cast_to_fp8(a_bf16)
        b_fp8, b_scale = per_block_cast_to_fp8(b_bf16.t())
        a_tensors.append(a_fp8)
        b_tensors.append(b_fp8)
        a_scales_tensors.append(a_scale)
        b_scales_tensors.append(b_scale)

    total_m = expert_offsets[-1].item()

    a_stack = torch.empty((total_m, k_g), device=device, dtype=torch.float8_e4m3fn)
    b_stack = torch.empty((num_groups, n_g, k_g), device=device, dtype=torch.float8_e4m3fn)
    a_bf16_stack = torch.empty((total_m, k_g), device=device, dtype=torch.bfloat16)
    b_bf16_stack = torch.empty((num_groups, n_g, k_g), device=device, dtype=torch.bfloat16)

    for g in range(num_groups):
        start = expert_offsets[g].item()
        end = expert_offsets[g + 1].item()
        a_stack[start:end] = a_tensors[g]
        b_stack[g] = b_tensors[g].t()  # [k_g, n_g] -> [n_g, k_g]
        a_bf16_stack[start:end] = a_bf16_tensors[g]
        b_bf16_stack[g] = b_bf16_tensors[g]

    # Non-contiguous transpose view — DO NOT call .contiguous() (see docstring)
    b_stack = b_stack.transpose(1, 2)

    a_scale_stack = torch.empty((total_m, k_g // 128), device=device, dtype=torch.float32)
    b_scale_stack = torch.empty((num_groups, n_g // 128, k_g // 128), device=device, dtype=torch.float32)

    for g in range(num_groups):
        start = expert_offsets[g].item()
        end = expert_offsets[g + 1].item()
        a_scale_stack[start:end] = a_scales_tensors[g]
        b_scale_stack[g] = b_scales_tensors[g].t()
    # Non-contiguous transpose view — must match b_stack layout
    b_scale_stack = b_scale_stack.transpose(1, 2)

    c_out = torch.empty((total_m, n_g), device=device, dtype=out_dtype)
    a_strides = torch.full((num_groups,), a_stack.stride(0), device=device, dtype=torch.int64)
    c_strides = torch.full((num_groups,), c_out.stride(0), device=device, dtype=torch.int64)

    # Pointer arrays — the kernel may or may not use these depending on the
    # code path; we populate them correctly regardless.
    a_ptrs = torch.empty((num_groups,), device=device, dtype=torch.int64)
    b_ptrs = torch.empty((num_groups,), device=device, dtype=torch.int64)
    out_ptrs = torch.empty((num_groups,), device=device, dtype=torch.int64)
    a_scales_ptrs = torch.empty((num_groups,), device=device, dtype=torch.int64)
    b_scales_ptrs = torch.empty((num_groups,), device=device, dtype=torch.int64)

    workspace = torch.empty((128 * 1024 * 1024), device=device, dtype=torch.uint8)

    def run_fn():
        fp8_blockwise_scaled_grouped_mm(
            c_out, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs,
            a_stack, b_stack, a_scale_stack, b_scale_stack,
            a_strides, a_strides, c_strides,
            layout_sfa, layout_sfb, problem_sizes,
            expert_offsets[:-1], workspace,
        )

    # Correctness check
    run_fn()
    torch.cuda.synchronize()
    fp8_output = c_out.clone()
    ref_output = compute_bf16_reference(
        n_g, k_g, num_groups, a_bf16_stack, b_bf16_stack, expert_offsets
    )
    acc = accuracy_metrics(fp8_output, ref_output)

    avg_time_us, avg_power_w = run_benchmark_loop(run_fn, num_warmup, num_run)

    return compute_metrics(avg_time_us, avg_power_w, total_m, n_g, k_g, acc)


# ---------------------------------------------------------------------------
# FP4 Grouped GEMM Benchmark
# ---------------------------------------------------------------------------
def bench_fp4(
    expected_m_per_group: int,
    n: int,
    k: int,
    num_groups: int,
    num_warmup: int,
    num_run: int,
) -> Dict:
    """
    FP4 grouped MoE GEMM benchmark using cutlass_fp4_group_mm.

    FP4 layout:
    - Weights: [e, n, k//2] as uint8 (two FP4 E2M1 values packed per byte)
    - Weight block scales: [e, n, k//16] as float8_e4m3fn (block size = 16)
    - Activations: same FP4 packed format via scaled_fp4_quant
    - Alphas: [e] as float32 = 1/(a_global_scale * w_global_scale)
    """
    device = "cuda"
    out_dtype = torch.bfloat16

    fp4_block_size = 16
    k_aligned = align_up(k, NK_ALIGNMENT)
    n_aligned = align_up(n, NK_ALIGNMENT)

    m_g = align_up(expected_m_per_group, M_ALIGNMENT)

    group_ms = [m_g for _ in range(num_groups)]

    # Build expert offsets
    expert_offsets = torch.zeros((num_groups + 1), device=device, dtype=torch.int32)
    for g in range(num_groups):
        expert_offsets[g + 1] = expert_offsets[g] + group_ms[g]
    total_m = expert_offsets[-1].item()

    # Build blockscale offsets
    blockscale_offsets = torch.zeros((num_groups + 1), device=device, dtype=torch.int32)
    for g in range(num_groups):
        blockscale_offsets[g + 1] = blockscale_offsets[g] + group_ms[g]

    # Problem sizes: [e, 3] -> (m_g, n_aligned, k_aligned)
    problem_sizes = torch.zeros((num_groups, 3), device=device, dtype=torch.int32)
    for g in range(num_groups):
        problem_sizes[g][:] = torch.tensor(
            [group_ms[g], n_aligned, k_aligned], device=device
        )

    # Generate BF16 data for reference and FP4 quantization
    a_bf16_list = []
    b_bf16_list = []
    a_fp4_list = []
    a_blockscale_list = []
    b_fp4_list = []
    b_blockscale_list = []
    a_global_scales = []
    b_global_scales = []

    for g in range(num_groups):
        a_bf16 = torch.randn((m_g, k_aligned), device=device, dtype=out_dtype)
        b_bf16 = torch.randn((n_aligned, k_aligned), device=device, dtype=out_dtype)

        a_bf16_list.append(a_bf16.clone())
        b_bf16_list.append(b_bf16.clone())

        a_gscale = (
            (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
            / a_bf16.flatten().abs().amax().clamp(1e-4)
        ).to(torch.float32)
        b_gscale = (
            (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
            / b_bf16.flatten().abs().amax().clamp(1e-4)
        ).to(torch.float32)

        a_global_scales.append(a_gscale)
        b_global_scales.append(b_gscale)

        a_fp4, a_bscale = scaled_fp4_quant(a_bf16, a_gscale)
        b_fp4, b_bscale = scaled_fp4_quant(b_bf16, b_gscale)

        a_fp4_list.append(a_fp4)
        a_blockscale_list.append(a_bscale)
        b_fp4_list.append(b_fp4)
        b_blockscale_list.append(b_bscale)

    # Stack activations contiguously
    a_fp4_stack = torch.empty(
        (total_m, k_aligned // 2), device=device, dtype=torch.uint8
    )
    a_blockscale_stack = torch.empty(
        (total_m, k_aligned // fp4_block_size),
        device=device,
        dtype=torch.float8_e4m3fn,
    )

    # Stack BF16 references
    a_bf16_stack = torch.empty(
        (total_m, k_aligned), device=device, dtype=out_dtype
    )
    b_bf16_stack = torch.empty(
        (num_groups, n_aligned, k_aligned), device=device, dtype=out_dtype
    )

    # Weights per expert
    b_fp4_stack = torch.empty(
        (num_groups, n_aligned, k_aligned // 2), device=device, dtype=torch.uint8
    )
    b_blockscale_stack = torch.empty(
        (num_groups, n_aligned, k_aligned // fp4_block_size),
        device=device,
        dtype=torch.float8_e4m3fn,
    )

    for g in range(num_groups):
        start = expert_offsets[g].item()
        end = expert_offsets[g + 1].item()
        a_fp4_stack[start:end] = a_fp4_list[g]
        a_blockscale_stack[start:end] = a_blockscale_list[g]
        b_fp4_stack[g] = b_fp4_list[g]
        b_blockscale_stack[g] = b_blockscale_list[g]
        a_bf16_stack[start:end] = a_bf16_list[g]
        b_bf16_stack[g] = b_bf16_list[g]

    # Per-expert alphas
    alphas = torch.empty((num_groups,), device=device, dtype=torch.float32)
    for g in range(num_groups):
        alphas[g] = 1.0 / (a_global_scales[g] * b_global_scales[g])

    ab_strides = torch.full(
        (num_groups,), k_aligned, device=device, dtype=torch.int64
    )
    c_strides = torch.full(
        (num_groups,), n_aligned, device=device, dtype=torch.int64
    )

    c_out = torch.empty((total_m, n_aligned), device=device, dtype=out_dtype)

    def run_fn():
        torch.ops.sgl_kernel.cutlass_fp4_group_mm.default(
            c_out,
            a_fp4_stack,
            b_fp4_stack,
            a_blockscale_stack,
            b_blockscale_stack,
            alphas,
            ab_strides,
            c_strides,
            problem_sizes,
            expert_offsets[:-1],
            blockscale_offsets[:-1],
        )

    # Correctness check
    run_fn()
    torch.cuda.synchronize()
    fp4_output = c_out.clone()

    ref_output = compute_bf16_reference(
        n_aligned, k_aligned, num_groups,
        a_bf16_stack, b_bf16_stack, expert_offsets,
    )
    acc = accuracy_metrics(fp4_output, ref_output)

    avg_time_us, avg_power_w = run_benchmark_loop(run_fn, num_warmup, num_run)

    return compute_metrics(avg_time_us, avg_power_w, total_m, n_aligned, k_aligned, acc)


benchmark_kernels = {
    "bf16_baseline": bench_bf16,
    "fp8_grouped": bench_fp8,
    "fp4_grouped": bench_fp4,
}


@dataclass
class ShapeArg:
    expected_m_per_group: int
    n: int
    k: int
    num_groups: int


def benchmark_one_shape(
    shape_args: List[ShapeArg],
    num_warmup: int,
    num_run: int,
    kernels_to_run: List[str],
):
    results = []

    for shape in shape_args:
        n_g = align_up(shape.n, NK_ALIGNMENT)
        k_g = align_up(shape.k, NK_ALIGNMENT)
        m_g = align_up(shape.expected_m_per_group, M_ALIGNMENT)
        print(
            f"\n{'='*80}\n"
            f"Benchmark: expected_m_per_group={shape.expected_m_per_group} "
            f"(aligned={m_g}), "
            f"n={shape.n} (aligned={n_g}), "
            f"k={shape.k} (aligned={k_g}), "
            f"num_groups={shape.num_groups}\n"
            f"{'='*80}"
        )
        for kernel_name in kernels_to_run:
            kernel_func = benchmark_kernels[kernel_name]
            try:
                metrics = kernel_func(
                    shape.expected_m_per_group,
                    shape.n,
                    shape.k,
                    shape.num_groups,
                    num_warmup,
                    num_run,
                )

                print(f"\n  Kernel: {kernel_name}")
                print(f"  Total M (across groups):  {metrics['total_m']}")
                print(f"  Aligned N={metrics['n_aligned']}, K={metrics['k_aligned']}")
                print(f"  ---- Performance ----")
                print(f"  Latency:                  {metrics['latency_us']:.2f} us")
                print(f"  Throughput:               {metrics['tflops']:.2f} TFLOPS")
                if metrics["avg_power_w"] > 0:
                    print(f"  ---- Energy ----")
                    print(f"  Avg Power:                {metrics['avg_power_w']:.1f} W")
                    print(f"  Energy per call:          {metrics['energy_mj']:.3f} mJ")
                    print(
                        f"  Energy Efficiency:        {metrics['tflops_per_watt']:.4f} TFLOPS/W"
                    )
                else:
                    print(f"  ---- Energy ----")
                    print(f"  (pynvml not available, energy metrics skipped)")
                print(f"  ---- Accuracy (vs BF16 reference) ----")
                print(
                    f"  Cosine Similarity:         {metrics['cosine_similarity']:.6f}"
                )
                print(
                    f"  Max Abs Error:             {metrics['max_abs_error']:.6f}"
                )
                print(
                    f"  Relative RMSE:             {metrics['relative_rmse']:.6f}"
                )

                results.append(
                    {
                        "m_per_group": shape.expected_m_per_group,
                        "n": shape.n,
                        "k": shape.k,
                        "num_groups": shape.num_groups,
                        "kernel": kernel_name,
                        **metrics,
                    }
                )
            except Exception as e:
                print(f"\n  Kernel: {kernel_name} — FAILED: {e}")

    print(f"\n\n{'='*110}")
    print("SUMMARY TABLE")
    print(f"{'='*110}")

    header = (
        f"{'Kernel':>15s} | {'Shape':>15s} | {'Latency(us)':>12s} | "
        f"{'TFLOPS':>8s} | "
    )
    if HAS_NVML:
        header += f"{'Power(W)':>9s} | {'TFLOPS/W':>10s} | "
    header += f"{'CosSim':>10s} | {'MaxAbsErr':>10s} | {'RelRMSE':>10s}"
    print(header)
    print("-" * len(header))

    for r in results:
        shape_str = f"m={r['m_per_group']},n={r['n']},g={r['num_groups']}"
        row = (
            f"{r['kernel']:>15s} | {shape_str:>25s} | "
            f"{r['latency_us']:>12.2f} | {r['tflops']:>8.2f} | "
        )
        if HAS_NVML:
            row += f"{r['avg_power_w']:>9.1f} | {r['tflops_per_watt']:>10.4f} | "
        row += (
            f"{r['cosine_similarity']:>10.6f} | "
            f"{r['max_abs_error']:>10.4f} | "
            f"{r['relative_rmse']:>10.6f}"
        )
        print(row)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Grouped GEMM benchmark: BF16 (bmm) vs FP8 vs FP4"
    )
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-run", type=int, default=2000)
    parser.add_argument(
        "--kernels",
        nargs="+",
        type=str,
        default=["bf16_baseline", "fp8_grouped", "fp4_grouped"],
        choices=list(benchmark_kernels.keys()),
        help="Which kernels to benchmark",
    )

    if IS_CI:
        shape_args = [
            ShapeArg(expected_m_per_group=128, n=512, k=7168, num_groups=256),
        ]
    else:
        shape_args = [
            # Prefill, DeepSeek-R1, gateup, chunk_size = 4096, TP = 8
            ShapeArg(expected_m_per_group=128, n=512, k=7168, num_groups=256),
            # Prefill, DeepSeek-R1, gateup, chunk_size = 8192, TP = 8
            ShapeArg(expected_m_per_group=256, n=512, k=7168, num_groups=256),
            # Prefill, DeepSeek-R1, gateup, chunk_size = 8192, TP = 16
            ShapeArg(expected_m_per_group=256, n=256, k=7168, num_groups=256),
            # Prefill, DeepSeek-R1, gateup, chunk_size = 16384, TP = 16
            ShapeArg(expected_m_per_group=512, n=256, k=7168, num_groups=256),
            # Decode, DeepSeek-R1, gateup, bs = 32, TP = 8
            ShapeArg(expected_m_per_group=1, n=512, k=7168, num_groups=256),
            # Decode, DeepSeek-R1, gateup, bs = 64, TP = 16
            ShapeArg(expected_m_per_group=2, n=256, k=7168, num_groups=256),
            # Prefill, DeepSeek-R1, gateup, chunk_size = 8192, EP = 8
            ShapeArg(expected_m_per_group=256, n=4096, k=7168, num_groups=32),
            # Prefill, DeepSeek-R1, gateup, chunk_size = 16384, EP = 16
            ShapeArg(expected_m_per_group=512, n=4096, k=7168, num_groups=16),
            # Decode, DeepSeek-R1, gateup, bs = 128, EP = 8
            ShapeArg(expected_m_per_group=4, n=4096, k=7168, num_groups=32),
            # Decode, DeepSeek-R1, gateup, bs = 256, EP = 16
            ShapeArg(expected_m_per_group=8, n=4096, k=7168, num_groups=16),
            # Prefill, Qwen3-235B-A22B-FP8, gateup, chunk_size = 16384, TP = 4
            ShapeArg(expected_m_per_group=1024, n=768, k=4096, num_groups=128),
            # Prefill, Qwen3-235B-A22B-FP8, down, chunk_size = 16384, TP = 4
            ShapeArg(expected_m_per_group=1024, n=4096, k=384, num_groups=128),
            # Decode, Qwen3-235B-A22B-FP8, gateup, bs = 256, TP = 4
            ShapeArg(expected_m_per_group=16, n=768, k=4096, num_groups=128),
            # Decode, Qwen3-235B-A22B-FP8, down, bs = 256, TP = 4
            ShapeArg(expected_m_per_group=16, n=4096, k=384, num_groups=128),
            # 1. Decode, TP — extreme bandwidth-bound (tiny M, small N, many experts)
            # ShapeArg(expected_m_per_group=1, n=512, k=7168, num_groups=256),
            # 2. Prefill, TP — compute-bound (large M, small N, many experts)
            # ShapeArg(expected_m_per_group=256, n=512, k=7168, num_groups=256),
            # 3. Decode, EP — bandwidth-bound but larger expert
            # ShapeArg(expected_m_per_group=4, n=4096, k=7168, num_groups=32),
            # 4. Prefill, EP — compute-bound with large expert
            # ShapeArg(expected_m_per_group=512, n=4096, k=7168, num_groups=16),
        ]
    args = parser.parse_args()
    benchmark_one_shape(shape_args, args.num_warmup, args.num_run, args.kernels)


if __name__ == "__main__":
    main()