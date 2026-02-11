import argparse
import os
import threading
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
from sgl_kernel import fp8_blockwise_scaled_grouped_mm
from sgl_kernel import cutlass_fp4_group_mm, scaled_fp4_quant

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_NVML = True
except Exception:
    HAS_NVML = False
    print("WARNING: pynvml not available. Energy metrics will be skipped.")
    print("Install with: pip install nvidia-ml-py\n")

M_ALIGNMENT = 16
NK_ALIGNMENT = 16


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
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
        x_view.size(0), x_view.size(2)
    )


class PowerSampler:
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


@dataclass
class BenchmarkData:
    """Holds shared input data and reference output for fairness."""
    a_bf16: torch.Tensor  # [total_m, k_aligned]
    b_bf16: torch.Tensor  # [num_groups, n_aligned, k_aligned]
    expert_offsets: torch.Tensor  # [num_groups + 1]
    ref_out: torch.Tensor  # [total_m, n_aligned]
    m_per_group: int
    n: int
    k: int
    num_groups: int


def generate_benchmark_data(m_per_group: int, n: int, k: int, num_groups: int) -> BenchmarkData:
    device = "cuda"
    m_g = align_up(m_per_group, M_ALIGNMENT)
    n_g = align_up(n, NK_ALIGNMENT)
    k_g = align_up(k, NK_ALIGNMENT)

    expert_offsets = torch.zeros((num_groups + 1), device=device, dtype=torch.int32)
    for g in range(num_groups):
        expert_offsets[g + 1] = expert_offsets[g] + m_g
    total_m = expert_offsets[-1].item()

    a_bf16 = torch.randn((total_m, k_g), device=device, dtype=torch.bfloat16)
    b_bf16 = torch.randn((num_groups, n_g, k_g), device=device, dtype=torch.bfloat16)

    a_reshaped = a_bf16.view(num_groups, m_g, k_g)
    b_transposed = b_bf16.transpose(1, 2)  # [g, k, n]

    c_reshaped = torch.bmm(a_reshaped, b_transposed)
    ref_out = c_reshaped.view(total_m, n_g)

    return BenchmarkData(
        a_bf16=a_bf16,
        b_bf16=b_bf16,
        expert_offsets=expert_offsets,
        ref_out=ref_out,
        m_per_group=m_per_group,
        n=n,
        k=k,
        num_groups=num_groups
    )


def accuracy_metrics(test: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    t = test.float().flatten()
    r = ref.float().flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        t.unsqueeze(0), r.unsqueeze(0)
    ).item()
    abs_err = (t - r).abs()
    max_abs_err = abs_err.max().item()
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

    avg_time_us = start_event.elapsed_time(end_event) / num_run * 1000
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


def bench_bf16(data: BenchmarkData, num_warmup: int, num_run: int) -> Dict:
    device = "cuda"
    m_g = align_up(data.m_per_group, M_ALIGNMENT)
    n_g = align_up(data.n, NK_ALIGNMENT)
    k_g = align_up(data.k, NK_ALIGNMENT)

    a_view = data.a_bf16.view(data.num_groups, m_g, k_g)
    b_view = data.b_bf16.transpose(1, 2).contiguous()  # [g, n, k] -> [g, k, n]

    c_out = torch.empty((data.num_groups, m_g, n_g), device=device, dtype=torch.bfloat16)

    def run_fn():
        torch.bmm(a_view, b_view, out=c_out)

    acc = {"cosine_similarity": 1.0, "max_abs_error": 0.0, "relative_rmse": 0.0}

    avg_time_us, avg_power_w = run_benchmark_loop(run_fn, num_warmup, num_run)
    return compute_metrics(avg_time_us, avg_power_w, data.a_bf16.shape[0], n_g, k_g, acc)


def bench_fp8(data: BenchmarkData, num_warmup: int, num_run: int) -> Dict:
    device = "cuda"
    m_g = align_up(data.m_per_group, M_ALIGNMENT)
    n_g = align_up(data.n, NK_ALIGNMENT)
    k_g = align_up(data.k, NK_ALIGNMENT)
    total_m = data.a_bf16.shape[0]

    a_fp8, a_scale = per_token_cast_to_fp8(data.a_bf16)
    b_tensors_fp8 = []
    b_scales_fp8 = []

    for g in range(data.num_groups):
        b_g = data.b_bf16[g]  # [n, k]
        b_fp8_g, b_scale_g = per_block_cast_to_fp8(b_g.t())  # cast [k, n]
        b_tensors_fp8.append(b_fp8_g)        # [k, n] in fp8
        b_scales_fp8.append(b_scale_g)        # scale for [k, n] blocks

    b_stack = torch.stack([t.t().contiguous() for t in b_tensors_fp8]) 
    b_scale_stack = torch.stack(
        [s.t().contiguous() for s in b_scales_fp8]
    )  # [g, scale_n, scale_k], contiguous

    # each group has the same [m, n, k]
    problem_sizes = torch.zeros((data.num_groups, 3), device=device, dtype=torch.int32)
    for g in range(data.num_groups):
        problem_sizes[g][:] = torch.tensor([m_g, n_g, k_g], device=device)

    a_scale_rows = m_g
    a_scale_cols = ceil_div(k_g, 128)
    layout_sfa = torch.zeros((data.num_groups, 5), device=device, dtype=torch.int32)
    for g in range(data.num_groups):
        layout_sfa[g] = torch.tensor(
            [g, a_scale_rows, a_scale_cols, a_scale_cols, 1], device=device
        )
    b_scale_rows = ceil_div(n_g, 128)
    b_scale_cols = ceil_div(k_g, 128)
    layout_sfb = torch.zeros((data.num_groups, 5), device=device, dtype=torch.int32)
    for g in range(data.num_groups):
        layout_sfb[g] = torch.tensor(
            [g, b_scale_rows, b_scale_cols, b_scale_cols, 1], device=device
        )

    c_out = torch.empty((total_m, n_g), device=device, dtype=torch.bfloat16)

    a_ptrs = torch.zeros(data.num_groups, device=device, dtype=torch.int64)
    b_ptrs = torch.zeros(data.num_groups, device=device, dtype=torch.int64)
    out_ptrs = torch.zeros(data.num_groups, device=device, dtype=torch.int64)
    a_scales_ptrs = torch.zeros(data.num_groups, device=device, dtype=torch.int64)
    b_scales_ptrs = torch.zeros(data.num_groups, device=device, dtype=torch.int64)

    for g in range(data.num_groups):
        start_row = data.expert_offsets[g].item()
        a_ptrs[g] = a_fp8[start_row].data_ptr()
        b_ptrs[g] = b_stack[g].data_ptr()
        out_ptrs[g] = c_out[start_row].data_ptr()
        a_scales_ptrs[g] = a_scale[start_row].data_ptr()
        b_scales_ptrs[g] = b_scale_stack[g].data_ptr()

    a_strides = torch.full((data.num_groups,), a_fp8.stride(0), device=device, dtype=torch.int64)
    c_strides = torch.full((data.num_groups,), c_out.stride(0), device=device, dtype=torch.int64)

    workspace = torch.empty((128 * 1024 * 1024), device=device, dtype=torch.uint8)

    def run_fn():
        fp8_blockwise_scaled_grouped_mm(
            c_out, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs,
            a_fp8, b_stack, a_scale, b_scale_stack,
            a_strides, a_strides, c_strides,
            layout_sfa, layout_sfb, problem_sizes,
            data.expert_offsets[:-1], workspace,
        )

    run_fn()
    torch.cuda.synchronize()

    acc = accuracy_metrics(c_out, data.ref_out)
    avg_time_us, avg_power_w = run_benchmark_loop(run_fn, num_warmup, num_run)

    return compute_metrics(avg_time_us, avg_power_w, total_m, n_g, k_g, acc)


def bench_fp4(data: BenchmarkData, num_warmup: int, num_run: int) -> Dict:
    device = "cuda"
    out_dtype = torch.bfloat16
    fp4_block_size = 16
    n_g = align_up(data.n, NK_ALIGNMENT)
    k_g = align_up(data.k, NK_ALIGNMENT)
    m_g = align_up(data.m_per_group, M_ALIGNMENT)

    a_fp4_list = []
    a_blockscale_list = []
    b_fp4_list = []
    b_blockscale_list = []
    a_global_scales = []
    b_global_scales = []

    for g in range(data.num_groups):
        start = data.expert_offsets[g].item()
        end = data.expert_offsets[g + 1].item()

        a_chunk = data.a_bf16[start:end]
        b_chunk = data.b_bf16[g]  # [n, k]

        a_gscale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / a_chunk.flatten().abs().amax().clamp(1e-4)).to(torch.float32)
        b_gscale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / b_chunk.flatten().abs().amax().clamp(1e-4)).to(torch.float32)

        a_global_scales.append(a_gscale)
        b_global_scales.append(b_gscale)

        a_fp4, a_bscale = scaled_fp4_quant(a_chunk, a_gscale)
        b_fp4, b_bscale = scaled_fp4_quant(b_chunk, b_gscale)

        a_fp4_list.append(a_fp4)
        a_blockscale_list.append(a_bscale)
        b_fp4_list.append(b_fp4)
        b_blockscale_list.append(b_bscale)

    total_m = data.a_bf16.shape[0]
    a_fp4_stack = torch.empty((total_m, k_g // 2), device=device, dtype=torch.uint8)
    a_blockscale_stack = torch.empty((total_m, k_g // fp4_block_size), device=device, dtype=torch.float8_e4m3fn)

    b_fp4_stack = torch.stack(b_fp4_list)
    b_blockscale_stack = torch.stack(b_blockscale_list)

    for g in range(data.num_groups):
        start = data.expert_offsets[g].item()
        end = data.expert_offsets[g + 1].item()
        a_fp4_stack[start:end] = a_fp4_list[g]
        a_blockscale_stack[start:end] = a_blockscale_list[g]

    alphas = torch.tensor([1.0 / (a * b) for a, b in zip(a_global_scales, b_global_scales)], device=device, dtype=torch.float32)

    group_ms = [m_g for _ in range(data.num_groups)]
    blockscale_offsets = torch.zeros((data.num_groups + 1), device=device, dtype=torch.int32)
    for g in range(data.num_groups):
        blockscale_offsets[g + 1] = blockscale_offsets[g] + group_ms[g]

    problem_sizes = torch.zeros((data.num_groups, 3), device=device, dtype=torch.int32)
    for g in range(data.num_groups):
        problem_sizes[g][:] = torch.tensor([group_ms[g], n_g, k_g], device=device)

    ab_strides = torch.full((data.num_groups,), k_g, device=device, dtype=torch.int64)
    c_strides = torch.full((data.num_groups,), n_g, device=device, dtype=torch.int64)

    params = {
        "ab_strides": ab_strides,
        "c_strides": c_strides,
        "problem_sizes": problem_sizes,
        "expert_offsets": data.expert_offsets[:-1],
        "blockscale_offsets": blockscale_offsets[:-1],
    }

    def run_fn():
        return cutlass_fp4_group_mm(
            a_fp4_stack, b_fp4_stack, a_blockscale_stack, b_blockscale_stack,
            alphas, out_dtype, device, params,
        )

    c_out = run_fn()
    torch.cuda.synchronize()
    acc = accuracy_metrics(c_out, data.ref_out)
    avg_time_us, avg_power_w = run_benchmark_loop(run_fn, num_warmup, num_run)
    return compute_metrics(avg_time_us, avg_power_w, total_m, n_g, k_g, acc)


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
        data = generate_benchmark_data(shape.expected_m_per_group, shape.n, shape.k, shape.num_groups)

        print(
            f"\n{'='*80}\n"
            f"Benchmark: expected_m_per_group={shape.expected_m_per_group} "
            f"(aligned={align_up(shape.expected_m_per_group, M_ALIGNMENT)}), "
            f"n={shape.n} (aligned={align_up(shape.n, NK_ALIGNMENT)}), "
            f"k={shape.k} (aligned={align_up(shape.k, NK_ALIGNMENT)}), "
            f"num_groups={shape.num_groups}\n"
            f"{'='*80}"
        )

        for kernel_name in kernels_to_run:
            kernel_func = benchmark_kernels[kernel_name]
            try:
                metrics = kernel_func(data, num_warmup, num_run)

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
                print(f"  Cosine Similarity:         {metrics['cosine_similarity']:.6f}")
                print(f"  Max Abs Error:             {metrics['max_abs_error']:.6f}")
                print(f"  Relative RMSE:             {metrics['relative_rmse']:.6f}")

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
    parser.add_argument("--num-run", type=int, default=2000) # 5000 for results
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
        ]
    args = parser.parse_args()
    benchmark_one_shape(shape_args, args.num_warmup, args.num_run, args.kernels)


if __name__ == "__main__":
    main()