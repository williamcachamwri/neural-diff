#!/usr/bin/env python3
"""Capture actual host hardware, OS, storage, GPU, and software environment audit.

Produces timestamped JSON and human-readable log outputs under `evidence/environment/`
without fabricating any values. Every probe records actual stdout/stderr and exit codes.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_probe(cmd: list[str], timeout: float = 15.0) -> dict[str, Any]:
    """Run an external command safely and capture actual output and exit code."""
    executable = shutil.which(cmd[0])
    if executable is None:
        return {
            "command": cmd,
            "available": False,
            "executable": None,
            "returncode": None,
            "stdout": "",
            "stderr": f"{cmd[0]} not found on PATH",
        }
    try:
        completed = subprocess.run(
            [executable, *cmd[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "command": cmd,
            "available": completed.returncode == 0,
            "executable": executable,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {
            "command": cmd,
            "available": False,
            "executable": executable,
            "returncode": -1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def _detect_memory_bytes() -> dict[str, int | None]:
    """Detect physical memory using OS-native APIs without external dependencies."""
    if sys.platform == "win32":

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return {
                "total_bytes": int(stat.ullTotalPhys),
                "available_bytes": int(stat.ullAvailPhys),
            }
    elif hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        avail = (
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
            if "SC_AVPHYS_PAGES" in os.sysconf_names
            else None
        )
        return {"total_bytes": int(total), "available_bytes": int(avail) if avail is not None else None}
    return {"total_bytes": None, "available_bytes": None}


def _detect_storage() -> dict[str, Any]:
    """Inspect storage capacity for the repository drive and user home drive."""
    drives: dict[str, Any] = {}
    targets = {
        "workspace_drive": REPO_ROOT,
        "home_drive": Path.home(),
    }
    for label, path in targets.items():
        try:
            usage = shutil.disk_usage(path)
            drives[label] = {
                "path": str(path),
                "total_gb": round(usage.total / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
            }
        except OSError as exc:
            drives[label] = {"path": str(path), "error": str(exc)}
    return drives


def _probe_python_packages() -> dict[str, Any]:
    """Check Python packages required by NEURAL DIFF and run real GPU/MPS probes if torch exists."""
    packages = [
        "torch",
        "transformers",
        "safetensors",
        "numpy",
        "scipy",
        "pytest",
        "hypothesis",
        "ruff",
        "mypy",
        "fastapi",
        "pydantic",
        "uvicorn",
        "httpx",
        "psutil",
        "huggingface_hub",
        "playwright",
    ]
    results: dict[str, Any] = {}
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", None)
            if version is None:
                from importlib.metadata import version as meta_version

                try:
                    version = meta_version(pkg)
                except Exception:
                    version = "installed"
            entry: dict[str, Any] = {"installed": True, "version": str(version)}
            if pkg == "torch":
                cuda_avail = bool(mod.cuda.is_available())
                mps_avail = bool(
                    hasattr(mod.backends, "mps") and mod.backends.mps.is_available()
                )
                entry["cuda_available"] = cuda_avail
                entry["cuda_compiled_version"] = getattr(mod.version, "cuda", None)
                entry["mps_available"] = mps_avail
                if cuda_avail:
                    device_count = mod.cuda.device_count()
                    devices = []
                    for idx in range(device_count):
                        props = mod.cuda.get_device_properties(idx)
                        kernel_verified = False
                        kernel_error = None
                        try:
                            a = mod.tensor([1.0, 2.0, 3.0], device=f"cuda:{idx}")
                            b = mod.tensor([4.0, 5.0, 6.0], device=f"cuda:{idx}")
                            dot = float(mod.dot(a, b).item())
                            kernel_verified = abs(dot - 32.0) < 1e-5
                        except Exception as kexc:
                            kernel_error = f"{type(kexc).__name__}: {kexc}"
                        devices.append(
                            {
                                "index": idx,
                                "name": props.name,
                                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                                "compute_capability": f"{props.major}.{props.minor}",
                                "multi_processor_count": props.multi_processor_count,
                                "kernel_execution_verified": kernel_verified,
                                "kernel_error": kernel_error,
                            }
                        )
                    entry["cuda_devices"] = devices
            results[pkg] = entry
        except Exception as exc:
            results[pkg] = {
                "installed": False,
                "version": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return results


def collect_environment_audit(stage: str) -> dict[str, Any]:
    """Collect a full, factual snapshot of the machine and toolchain."""
    now_utc = dt.datetime.now(dt.UTC).isoformat()
    mem = _detect_memory_bytes()
    arch = platform.machine()
    is_apple_silicon = sys.platform == "darwin" and arch in ("arm64", "aarch64")

    cli_tools = {
        "python": _run_probe([sys.executable, "--version"]),
        "uv": _run_probe(["uv", "--version"]),
        "pip": _run_probe([sys.executable, "-m", "pip", "--version"]),
        "git": _run_probe(["git", "--version"]),
        "node": _run_probe(["node", "--version"]),
        "npm": _run_probe(["npm", "--version"]),
        "npx": _run_probe(["npx", "--version"]),
        "docker": _run_probe(["docker", "--version"]),
        "nvidia_smi": _run_probe(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
                "--format=csv,noheader",
            ]
        ),
    }

    frontend_pkg_json = REPO_ROOT / "frontend" / "package.json"
    frontend_packages: dict[str, Any] = {"package_json_present": frontend_pkg_json.exists()}
    if frontend_pkg_json.exists():
        try:
            pkg_data = json.loads(frontend_pkg_json.read_text(encoding="utf-8"))
            frontend_packages["dependencies"] = pkg_data.get("dependencies", {})
            frontend_packages["devDependencies"] = pkg_data.get("devDependencies", {})
        except Exception as exc:
            frontend_packages["error"] = str(exc)

    return {
        "captured_at_utc": now_utc,
        "stage": stage,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "architecture": arch,
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "apple_silicon_metal": {
            "apple_silicon_hardware": is_apple_silicon,
            "metal_mps_supported": False if not is_apple_silicon else None,
        },
        "memory": {
            "total_gb": round(mem["total_bytes"] / (1024**3), 2) if mem["total_bytes"] else None,
            "available_gb": (
                round(mem["available_bytes"] / (1024**3), 2) if mem["available_bytes"] else None
            ),
            **mem,
        },
        "storage": _detect_storage(),
        "python_runtime": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
            "meets_3_11_plus": sys.version_info >= (3, 11),
            "in_virtualenv": sys.prefix != sys.base_prefix,
            "prefix": sys.prefix,
        },
        "cli_tools": cli_tools,
        "python_packages": _probe_python_packages(),
        "frontend_packages": frontend_packages,
    }


def format_human_log(audit: dict[str, Any]) -> str:
    """Format the audit dictionary as a clear, verifiable terminal report."""
    lines = [
        "=" * 76,
        f"NEURAL DIFF ENVIRONMENT AUDIT — STAGE: {audit['stage']}",
        f"Captured (UTC): {audit['captured_at_utc']}",
        "=" * 76,
        "",
        "[OS & CPU]",
        f"  Platform       : {audit['os']['platform']}",
        f"  Architecture   : {audit['os']['architecture']}",
        f"  Processor      : {audit['os']['processor']}",
        f"  Logical CPUs   : {audit['os']['logical_cpu_count']}",
        f"  Apple Silicon  : {audit['apple_silicon_metal']['apple_silicon_hardware']}",
        "",
        "[MEMORY & STORAGE]",
        f"  Total RAM (GB) : {audit['memory']['total_gb']}",
        f"  Avail RAM (GB) : {audit['memory']['available_gb']}",
    ]
    for label, info in audit["storage"].items():
        if "total_gb" in info:
            lines.append(
                f"  {label:15s}: {info['path']} — {info['free_gb']} GB free / {info['total_gb']} GB total"
            )
    lines.extend(
        [
            "",
            "[PYTHON RUNTIME]",
            f"  Executable     : {audit['python_runtime']['executable']}",
            f"  Version        : {audit['python_runtime']['version'].splitlines()[0]}",
            f"  Python >= 3.11 : {audit['python_runtime']['meets_3_11_plus']}",
            f"  Virtualenv     : {audit['python_runtime']['in_virtualenv']}",
            "",
            "[CLI TOOLS]",
        ]
    )
    for name, probe in audit["cli_tools"].items():
        status = "AVAILABLE" if probe["available"] else "MISSING/UNAVAILABLE"
        detail = probe["stdout"] or probe["stderr"]
        lines.append(f"  {name:12s}: [{status}] rc={probe['returncode']} | {detail}")
    lines.extend(["", "[PYTHON PACKAGES]"])
    for pkg, info in audit["python_packages"].items():
        if info["installed"]:
            extra = ""
            if pkg == "torch":
                extra = (
                    f" (cuda_available={info.get('cuda_available')}, "
                    f"cuda_compiled={info.get('cuda_compiled_version')}, "
                    f"devices={info.get('cuda_devices', [])})"
                )
            lines.append(f"  {pkg:16s}: INSTALLED ({info['version']}){extra}")
        else:
            lines.append(f"  {pkg:16s}: NOT INSTALLED")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture NEURAL DIFF environment audit evidence.")
    parser.add_argument(
        "--stage",
        default="current",
        help="Audit stage label (e.g. baseline, post-install, current).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "evidence" / "environment"),
        help="Output directory for JSON and log artifacts.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = collect_environment_audit(stage=args.stage)
    log_text = format_human_log(audit)

    json_path = out_dir / f"environment_{args.stage}.json"
    log_path = out_dir / f"environment_{args.stage}.log"

    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    log_path.write_text(log_text, encoding="utf-8")

    print(log_text)
    print(f"Saved JSON audit : {json_path}")
    print(f"Saved Log audit  : {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
