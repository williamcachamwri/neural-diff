"""Hardware resource telemetry, device selection, cancellation, and memory cleanup."""

from __future__ import annotations

import gc
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from core.exceptions import OperationCancelledError, ResourceExhaustedError


class CancellationToken:
    """Thread-safe cooperative cancellation token for long-running diff or inference jobs."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason: str = "Operation cancelled by caller."
        self._lock = threading.Lock()

    def cancel(self, reason: str = "Operation cancelled by caller.") -> None:
        with self._lock:
            self._reason = reason
            self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def check(self) -> None:
        """Raise `OperationCancelledError` immediately if cancelled."""
        if self._event.is_set():
            raise OperationCancelledError(self.reason)


@dataclass
class ResourceSnapshot:
    """Measured process and GPU memory usage at a point in time."""

    timestamp_monotonic: float
    process_rss_mb: float
    system_available_ram_mb: float
    system_total_ram_mb: float
    cuda_available: bool
    cuda_allocated_mb: float
    cuda_reserved_mb: float
    cuda_total_vram_mb: float
    selected_device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResourceManager:
    """Manages CPU/GPU device selection, memory guardrails, and deterministic cleanup."""

    def __init__(
        self,
        preferred_device: str = "auto",
        min_free_ram_mb: float = 256.0,
        min_free_vram_mb: float = 512.0,
        chunk_elements: int = 1_048_576,
    ) -> None:
        self.preferred_device = preferred_device
        self.min_free_ram_mb = min_free_ram_mb
        self.min_free_vram_mb = min_free_vram_mb
        self.chunk_elements = max(4096, int(chunk_elements))
        self._resolved_device: str | None = None
        self._device_reason: str = ""

    def resolve_device(self) -> str:
        """Resolve execution device (`cuda:0`, `mps`, or `cpu`) with real kernel verification."""
        if self._resolved_device is not None:
            return self._resolved_device

        if self.preferred_device == "cpu":
            self._resolved_device = "cpu"
            self._device_reason = "Explicitly configured for CPU execution."
            return "cpu"

        try:
            import torch

            if self.preferred_device in ("auto", "cuda") or self.preferred_device.startswith("cuda"):
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    target = "cuda:0" if self.preferred_device in ("auto", "cuda") else self.preferred_device
                    idx = int(target.split(":")[1]) if ":" in target else 0
                    free_bytes, _total_bytes = torch.cuda.mem_get_info(idx)
                    free_mb = free_bytes / (1024**2)
                    if free_mb >= self.min_free_vram_mb:
                        # Execute actual tensor op to confirm architecture kernel availability (e.g. sm_120)
                        probe = torch.tensor([1.0, 2.0], device=target)
                        if float(torch.sum(probe).item()) == 3.0:
                            self._resolved_device = target
                            self._device_reason = (
                                f"Verified CUDA kernel execution on {torch.cuda.get_device_name(idx)} "
                                f"({free_mb:.0f} MB free VRAM)."
                            )
                            return target
                    else:
                        self._device_reason = (
                            f"CUDA free VRAM ({free_mb:.0f} MB) below threshold ({self.min_free_vram_mb:.0f} MB); falling back to CPU."
                        )

            if self.preferred_device in ("auto", "mps"):
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._resolved_device = "mps"
                    self._device_reason = "Apple Metal (MPS) backend available."
                    return "mps"
        except Exception as exc:
            self._device_reason = f"GPU probe failed ({type(exc).__name__}: {exc}); falling back to CPU."

        if not self._device_reason:
            self._device_reason = "No accelerated GPU available; using CPU."
        self._resolved_device = "cpu"
        return "cpu"

    @property
    def device_reason(self) -> str:
        if self._resolved_device is None:
            self.resolve_device()
        return self._device_reason

    def snapshot(self) -> ResourceSnapshot:
        """Capture current process RSS, system RAM, and CUDA VRAM metrics."""
        import psutil

        proc = psutil.Process()
        rss_mb = proc.memory_info().rss / (1024**2)
        vm = psutil.virtual_memory()
        avail_ram_mb = vm.available / (1024**2)
        total_ram_mb = vm.total / (1024**2)

        cuda_avail = False
        cuda_alloc_mb = 0.0
        cuda_res_mb = 0.0
        cuda_total_mb = 0.0
        try:
            import torch

            cuda_avail = bool(torch.cuda.is_available())
            if cuda_avail and torch.cuda.device_count() > 0:
                cuda_alloc_mb = torch.cuda.memory_allocated(0) / (1024**2)
                cuda_res_mb = torch.cuda.memory_reserved(0) / (1024**2)
                cuda_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        except Exception:
            pass

        return ResourceSnapshot(
            timestamp_monotonic=time.perf_counter(),
            process_rss_mb=round(rss_mb, 2),
            system_available_ram_mb=round(avail_ram_mb, 2),
            system_total_ram_mb=round(total_ram_mb, 2),
            cuda_available=cuda_avail,
            cuda_allocated_mb=round(cuda_alloc_mb, 2),
            cuda_reserved_mb=round(cuda_res_mb, 2),
            cuda_total_vram_mb=round(cuda_total_mb, 2),
            selected_device=self.resolve_device(),
        )

    def check_memory_headroom(self) -> ResourceSnapshot:
        """Verify available host RAM meets safety minimum; attempt GC first if tight."""
        snap = self.snapshot()
        if snap.system_available_ram_mb < self.min_free_ram_mb:
            self.cleanup()
            snap = self.snapshot()
            if snap.system_available_ram_mb < self.min_free_ram_mb:
                raise ResourceExhaustedError(
                    f"Available system RAM ({snap.system_available_ram_mb:.1f} MB) is below safety floor ({self.min_free_ram_mb:.1f} MB).",
                    details=snap.to_dict(),
                )
        return snap

    @staticmethod
    def cleanup() -> None:
        """Deterministically run Python GC and release unoccupied CUDA cached blocks."""
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @contextmanager
    def managed_execution(
        self, token: CancellationToken | None = None
    ) -> Iterator[ dict[str, Any] ]:
        """Context manager recording start/end timing, peak memory delta, and cleanup."""
        if token is not None:
            token.check()
        before = self.snapshot()
        t0 = time.perf_counter()
        telemetry: dict[str, Any] = {"before": before.to_dict()}
        try:
            yield telemetry
        finally:
            elapsed = time.perf_counter() - t0
            after_pre_gc = self.snapshot()
            self.cleanup()
            after_post_gc = self.snapshot()
            telemetry["after_pre_cleanup"] = after_pre_gc.to_dict()
            telemetry["after_post_cleanup"] = after_post_gc.to_dict()
            telemetry["duration_seconds"] = round(elapsed, 6)
            telemetry["rss_delta_mb"] = round(
                after_pre_gc.process_rss_mb - before.process_rss_mb, 2
            )
            telemetry["cuda_allocated_peak_delta_mb"] = round(
                after_pre_gc.cuda_allocated_mb - before.cuda_allocated_mb, 2
            )
