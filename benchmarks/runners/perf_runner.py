"""Throughput and memory micro-benchmarks for chunked tensor processing (Phase 16)."""

from __future__ import annotations

import datetime as dt
import tempfile
import time
from pathlib import Path
from typing import Any

import safetensors.torch
import torch

from core.loaders.model_loader import ModelLoader
from core.resource_manager import ResourceManager
from core.security import SecurityPolicy
from core.tensor_engine.comparator import TensorDiffEngine
from core.tensor_engine.metrics import DiffTolerances, compute_chunked_tensor_metrics


def run_chunked_tensor_microbenchmarks() -> dict[str, Any]:
    """Measure real wall-clock throughput and RSS footprint across tensor & chunk sizes."""
    rm = ResourceManager(preferred_device="auto")
    measurements: list[dict[str, Any]] = []

    configs = [
        ("1M_float32_chunk64k", 1_000_000, torch.float32, 65_536),
        ("1M_float32_chunk1M", 1_000_000, torch.float32, 1_048_576),
        ("10M_float32_chunk256k", 10_000_000, torch.float32, 262_144),
        ("10M_float32_chunk1M", 10_000_000, torch.float32, 1_048_576),
        ("10M_bfloat16_chunk1M", 10_000_000, torch.bfloat16, 1_048_576),
        ("25M_bfloat16_chunk2M", 25_000_000, torch.bfloat16, 2_097_152),
    ]

    for label, num_elements, dtype, chunk_elems in configs:
        g1 = torch.Generator().manual_seed(11)
        g2 = torch.Generator().manual_seed(22)
        ta = torch.randn(num_elements, generator=g1).to(dtype)
        tb = (ta.to(torch.float32) + torch.randn(num_elements, generator=g2) * 0.01).to(dtype)
        bytes_compared = num_elements * ta.element_size() * 2

        rm.cleanup()
        snap_before = rm.snapshot()
        t0 = time.perf_counter()
        metrics = compute_chunked_tensor_metrics(
            [ta],
            [tb],
            tolerances=DiffTolerances(),
            chunk_elements=chunk_elems,
        )
        elapsed = time.perf_counter() - t0
        snap_after = rm.snapshot()

        elems_per_sec = num_elements / max(elapsed, 1e-9)
        gb_per_sec = (bytes_compared / (1024**3)) / max(elapsed, 1e-9)

        measurements.append(
            {
                "benchmark_id": label,
                "experiment_type": "SYNTHETIC FIXTURE EXPERIMENT",
                "num_elements": num_elements,
                "dtype": str(dtype).replace("torch.", ""),
                "chunk_elements": chunk_elems,
                "duration_seconds": round(elapsed, 6),
                "millions_elements_per_sec": round(elems_per_sec / 1e6, 3),
                "effective_gb_per_sec": round(gb_per_sec, 3),
                "rss_before_mb": snap_before.process_rss_mb,
                "rss_after_mb": snap_after.process_rss_mb,
                "rss_delta_mb": round(snap_after.process_rss_mb - snap_before.process_rss_mb, 2),
                "observed_cosine": metrics.cosine_similarity,
                "observed_rmse": metrics.rmse,
            }
        )
        del ta, tb
        rm.cleanup()

    # Lazy Safetensors streaming disk-to-diff benchmark
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        policy = SecurityPolicy(allowed_roots=[tdir])
        fa = tdir / "stream_a.safetensors"
        fb = tdir / "stream_b.safetensors"
        w_a = {f"layer.{i}.weight": torch.randn((512, 1024), dtype=torch.bfloat16) for i in range(8)}
        w_b = {k: (v.to(torch.float32) * 0.995 + 0.002).to(torch.bfloat16) for k, v in w_a.items()}
        safetensors.torch.save_file(w_a, str(fa))
        safetensors.torch.save_file(w_b, str(fb))
        del w_a, w_b
        rm.cleanup()

        loader = ModelLoader(policy=policy)
        snap_s0 = rm.snapshot()
        t_s0 = time.perf_counter()
        with loader.load(fa) as ha, loader.load(fb) as hb:
            rep = TensorDiffEngine(chunk_elements=262_144, include_distributions=True).compare(ha, hb)
        elapsed_stream = time.perf_counter() - t_s0
        snap_s1 = rm.snapshot()

        streaming_bench = {
            "benchmark_id": "safetensors_8x512x1024_bf16_lazy_stream",
            "experiment_type": "SYNTHETIC FIXTURE EXPERIMENT",
            "tensor_count": rep.compared_tensors,
            "total_parameters": rep.total_compared_parameters,
            "duration_seconds": round(elapsed_stream, 6),
            "rss_before_mb": snap_s0.process_rss_mb,
            "rss_after_mb": snap_s1.process_rss_mb,
            "rss_delta_mb": round(snap_s1.process_rss_mb - snap_s0.process_rss_mb, 2),
            "global_mean_cosine": rep.global_mean_cosine,
        }

    return {
        "captured_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "device_probed": rm.resolve_device(),
        "chunked_microbenchmarks": measurements,
        "safetensors_lazy_streaming": streaming_bench,
    }
