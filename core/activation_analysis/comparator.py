"""Activation Diff Engine (Module E) — deterministic forward-hook capture and comparison."""

from __future__ import annotations

import math
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from torch import nn

import safetensors
from core.activation_analysis.metrics import (
    ActivationLayerComparison,
    compare_activation_tensors,
    compute_linear_cka,
)
from core.exceptions import ModelCompatibilityError
from core.loaders.model_loader import ModelLoader
from core.resource_manager import CancellationToken, ResourceManager
from core.security import SecurityPolicy
from core.types import EpistemicStatus

DEFAULT_DIAGNOSTIC_PROMPTS: tuple[str, ...] = (
    "Explain in one concise sentence how gradient descent updates neural network weights.",
    "Return a JSON object with keys 'language' and 'year' for Python.",
    "Tính giá trị biểu thức: 15 * 4 + 8 =",
    "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return",
)

_BLOCK_INDEX_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks|layer)\.(\d+)$")
_ATTN_MODULE_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks|layer)\.(\d+)\.(?:self_attn|attn|attention)$")
_MLP_ACT_MODULE_RE = re.compile(
    r"(?:^|\.)(?:layers|h|blocks|layer)\.(\d+)\.(?:mlp|ffn)\.(?:act_fn|act|activation_fn)$"
)
_MLP_MODULE_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks|layer)\.(\d+)\.(?:mlp|ffn)$")


@dataclass
class CapturedPromptTrace:
    """Captured forward-pass tensors for a single prompt on a single model."""

    prompt: str
    token_ids: list[int]
    tokens: list[str]
    hidden_states: dict[int, torch.Tensor]
    attention_outputs: dict[int, torch.Tensor]
    mlp_intermediates: dict[int, torch.Tensor]
    final_token_logits: torch.Tensor | None


@dataclass
class PromptActivationDiff:
    """Full comparison result for one controlled prompt across Model A and Model B."""

    prompt: str
    token_ids_a: list[int]
    token_ids_b: list[int]
    tokens_a: list[str]
    tokens_b: list[str]
    tokens_identical: bool
    next_token_kl_divergence: float | None
    next_token_js_divergence: float | None
    top_tokens_a: list[dict[str, Any]]
    top_tokens_b: list[dict[str, Any]]
    hidden_state_comparisons: list[ActivationLayerComparison]
    attention_comparisons: list[ActivationLayerComparison]
    mlp_comparisons: list[ActivationLayerComparison]
    limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "token_ids_a": self.token_ids_a,
            "token_ids_b": self.token_ids_b,
            "tokens_a": self.tokens_a,
            "tokens_b": self.tokens_b,
            "tokens_identical": self.tokens_identical,
            "next_token_kl_divergence": self.next_token_kl_divergence,
            "next_token_js_divergence": self.next_token_js_divergence,
            "top_tokens_a": self.top_tokens_a,
            "top_tokens_b": self.top_tokens_b,
            "hidden_state_comparisons": [c.to_dict() for c in self.hidden_state_comparisons],
            "attention_comparisons": [c.to_dict() for c in self.attention_comparisons],
            "mlp_comparisons": [c.to_dict() for c in self.mlp_comparisons],
            "limitations": self.limitations,
        }


@dataclass
class ActivationDiffReport:
    """Aggregate Activation Diff report with complete reproducibility provenance."""

    epistemic_status: EpistemicStatus
    model_a_id: str
    model_b_id: str
    revision_a: str | None
    revision_b: str | None
    tokenizer_a_info: dict[str, Any]
    tokenizer_b_info: dict[str, Any]
    inference_parameters: dict[str, Any]
    hardware_telemetry: dict[str, Any]
    software_versions: dict[str, str]
    prompt_count: int
    mean_hidden_cosine_by_layer: dict[int, float | None]
    mean_hidden_cka_by_layer: dict[int, float | None]
    mean_mlp_cosine_by_layer: dict[int, float | None]
    mean_attn_cosine_by_layer: dict[int, float | None]
    mean_final_token_js_divergence: float | None
    cross_prompt_pooled_cka_final_layer: float | None
    prompt_diffs: list[PromptActivationDiff]
    cross_architecture_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "epistemic_status": self.epistemic_status.value,
            "provenance": {
                "model_a_id": self.model_a_id,
                "model_b_id": self.model_b_id,
                "revision_a": self.revision_a,
                "revision_b": self.revision_b,
                "tokenizer_a": self.tokenizer_a_info,
                "tokenizer_b": self.tokenizer_b_info,
                "inference_parameters": self.inference_parameters,
                "hardware": self.hardware_telemetry,
                "software_versions": self.software_versions,
            },
            "summary": {
                "prompt_count": self.prompt_count,
                "mean_hidden_cosine_by_layer": {
                    str(k): v for k, v in self.mean_hidden_cosine_by_layer.items()
                },
                "mean_hidden_cka_by_layer": {
                    str(k): v for k, v in self.mean_hidden_cka_by_layer.items()
                },
                "mean_mlp_cosine_by_layer": {
                    str(k): v for k, v in self.mean_mlp_cosine_by_layer.items()
                },
                "mean_attn_cosine_by_layer": {
                    str(k): v for k, v in self.mean_attn_cosine_by_layer.items()
                },
                "mean_final_token_js_divergence": self.mean_final_token_js_divergence,
                "cross_prompt_pooled_cka_final_layer": self.cross_prompt_pooled_cka_final_layer,
            },
            "prompt_diffs": [p.to_dict() for p in self.prompt_diffs],
            "cross_architecture_notes": self.cross_architecture_notes,
        }


def _extract_first_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
        return output.last_hidden_state
    return None


def capture_module_traces(
    module: nn.Module,
    tokenized_prompts: list[tuple[str, list[int], list[str]]],
    *,
    device: str = "cpu",
    seed: int = 42,
    cancellation_token: CancellationToken | None = None,
) -> list[CapturedPromptTrace]:
    """Run controlled deterministic forward passes on a PyTorch `nn.Module` with clean hook teardown."""
    module.eval()
    traces: list[CapturedPromptTrace] = []

    current_hidden: dict[int, torch.Tensor] = {}
    current_attn: dict[int, torch.Tensor] = {}
    current_mlp_act: dict[int, torch.Tensor] = {}
    current_mlp_out: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def _make_hook(bucket: dict[int, torch.Tensor], layer_idx: int):
        def _hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
            t = _extract_first_tensor(out)
            if t is not None:
                bucket[layer_idx] = t.detach().to(device="cpu", dtype=torch.float32)

        return _hook

    try:
        for name, submod in module.named_modules():
            if not name:
                continue
            m_block = _BLOCK_INDEX_RE.search(name)
            if m_block:
                handles.append(submod.register_forward_hook(_make_hook(current_hidden, int(m_block.group(1)))))
                continue
            m_attn = _ATTN_MODULE_RE.search(name)
            if m_attn:
                handles.append(submod.register_forward_hook(_make_hook(current_attn, int(m_attn.group(1)))))
                continue
            m_mlp_act = _MLP_ACT_MODULE_RE.search(name)
            if m_mlp_act:
                handles.append(submod.register_forward_hook(_make_hook(current_mlp_act, int(m_mlp_act.group(1)))))
                continue
            m_mlp = _MLP_MODULE_RE.search(name)
            if m_mlp:
                handles.append(submod.register_forward_hook(_make_hook(current_mlp_out, int(m_mlp.group(1)))))

        with torch.no_grad():
            for prompt_str, tok_ids, tok_strs in tokenized_prompts:
                if cancellation_token is not None:
                    cancellation_token.check()
                current_hidden.clear()
                current_attn.clear()
                current_mlp_act.clear()
                current_mlp_out.clear()

                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                input_ids = torch.tensor([tok_ids], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids, device=device)

                try:
                    out = module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                except TypeError:
                    out = module(input_ids)

                # Prefer hook-captured hidden states; fall back to HF `hidden_states` tuple if needed
                if not current_hidden and hasattr(out, "hidden_states") and out.hidden_states:
                    for idx, hs in enumerate(out.hidden_states[1:]):
                        current_hidden[idx] = hs.detach().to(device="cpu", dtype=torch.float32)

                mlp_effective = dict(current_mlp_act) if current_mlp_act else dict(current_mlp_out)

                final_logits: torch.Tensor | None = None
                if hasattr(out, "logits") and isinstance(out.logits, torch.Tensor):
                    final_logits = out.logits[0, -1, :].detach().to(device="cpu", dtype=torch.float32)
                elif isinstance(out, torch.Tensor) and out.ndim == 3:
                    final_logits = out[0, -1, :].detach().to(device="cpu", dtype=torch.float32)

                traces.append(
                    CapturedPromptTrace(
                        prompt=prompt_str,
                        token_ids=list(tok_ids),
                        tokens=list(tok_strs),
                        hidden_states=dict(current_hidden),
                        attention_outputs=dict(current_attn),
                        mlp_intermediates=mlp_effective,
                        final_token_logits=final_logits,
                    )
                )
    finally:
        for h in handles:
            h.remove()

    return traces


def _compute_logit_divergences(
    logits_a: torch.Tensor | None,
    logits_b: torch.Tensor | None,
    decode_a,
    decode_b,
) -> tuple[float | None, float | None, list[dict[str, Any]], list[dict[str, Any]]]:
    top_a: list[dict[str, Any]] = []
    top_b: list[dict[str, Any]] = []

    probs_a: torch.Tensor | None = None
    probs_b: torch.Tensor | None = None

    if logits_a is not None and logits_a.numel() > 0 and torch.isfinite(logits_a).all():
        probs_a = torch.softmax(logits_a.to(torch.float64), dim=-1)
        vals, idxs = torch.topk(probs_a, k=min(5, probs_a.numel()))
        for p, tid in zip(vals.tolist(), idxs.tolist(), strict=False):
            top_a.append({"token_id": int(tid), "token": decode_a(int(tid)), "prob": round(float(p), 6)})

    if logits_b is not None and logits_b.numel() > 0 and torch.isfinite(logits_b).all():
        probs_b = torch.softmax(logits_b.to(torch.float64), dim=-1)
        vals, idxs = torch.topk(probs_b, k=min(5, probs_b.numel()))
        for p, tid in zip(vals.tolist(), idxs.tolist(), strict=False):
            top_b.append({"token_id": int(tid), "token": decode_b(int(tid)), "prob": round(float(p), 6)})

    if probs_a is None or probs_b is None or probs_a.shape != probs_b.shape:
        return None, None, top_a, top_b

    eps = 1e-15
    pa = probs_a.clamp_min(eps)
    pb = probs_b.clamp_min(eps)
    m = 0.5 * (pa + pb)
    kl_ab = float((pa * (pa.log() - pb.log())).sum().item())
    js_ab = 0.5 * float((pa * (pa.log() - m.log())).sum().item()) + 0.5 * float(
        (pb * (pb.log() - m.log())).sum().item()
    )
    # Normalize JS divergence to bits (or [0, ln 2] nats -> clamp >= 0)
    return round(max(0.0, kl_ab), 8), round(max(0.0, js_ab), 8), top_a, top_b


class ActivationDiffEngine:
    """Memory-efficient sequential activation extractor and comparator for Hugging Face or PyTorch models."""

    def __init__(
        self,
        *,
        resource_manager: ResourceManager | None = None,
        policy: SecurityPolicy | None = None,
        seed: int = 42,
        max_prompt_tokens: int = 128,
        dtype: str = "float32",
    ) -> None:
        self.resource_manager = resource_manager or ResourceManager()
        self.policy = policy or SecurityPolicy()
        self.seed = seed
        self.max_prompt_tokens = max(8, int(max_prompt_tokens))
        self.dtype = dtype

    def compare_modules(
        self,
        module_a: nn.Module,
        module_b: nn.Module,
        tokenized_prompts_a: list[tuple[str, list[int], list[str]]],
        tokenized_prompts_b: list[tuple[str, list[int], list[str]]] | None = None,
        *,
        model_a_id: str = "module_a",
        model_b_id: str = "module_b",
        device: str = "cpu",
        cancellation_token: CancellationToken | None = None,
    ) -> ActivationDiffReport:
        """Compare two in-memory `nn.Module` instances on controlled token sequences."""
        prompts_b = tokenized_prompts_b if tokenized_prompts_b is not None else tokenized_prompts_a
        if len(tokenized_prompts_a) != len(prompts_b):
            raise ValueError("Prompt lists for Module A and Module B must have equal length.")

        with self.resource_manager.managed_execution(cancellation_token) as telemetry:
            traces_a = capture_module_traces(
                module_a.to(device),
                tokenized_prompts_a,
                device=device,
                seed=self.seed,
                cancellation_token=cancellation_token,
            )
            traces_b = capture_module_traces(
                module_b.to(device),
                prompts_b,
                device=device,
                seed=self.seed,
                cancellation_token=cancellation_token,
            )

        return self._build_report(
            traces_a=traces_a,
            traces_b=traces_b,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            revision_a=None,
            revision_b=None,
            tokenizer_a_info={"type": "direct_token_ids"},
            tokenizer_b_info={"type": "direct_token_ids"},
            decode_a=lambda tid: f"tok_{tid}",
            decode_b=lambda tid: f"tok_{tid}",
            device=device,
            telemetry=telemetry,
        )

    def compare_pretrained(
        self,
        source_a: str | Path,
        source_b: str | Path,
        prompts: list[str] | None = None,
        *,
        local_files_only: bool = False,
        cancellation_token: CancellationToken | None = None,
    ) -> ActivationDiffReport:
        """Load Model A and Model B sequentially (`trust_remote_code=False`), extract activations, and compare."""
        active_prompts = list(prompts) if prompts else list(DEFAULT_DIAGNOSTIC_PROMPTS)
        loader = ModelLoader(policy=self.policy)
        device = self.resource_manager.resolve_device()

        with loader.load(source_a, local_files_only=local_files_only) as handle_a:
            meta_a = handle_a.metadata
        with loader.load(source_b, local_files_only=local_files_only) as handle_b:
            meta_b = handle_b.metadata

        dir_a = Path(meta_a.resolved_path) if Path(meta_a.resolved_path).is_dir() else Path(meta_a.resolved_path).parent
        dir_b = Path(meta_b.resolved_path) if Path(meta_b.resolved_path).is_dir() else Path(meta_b.resolved_path).parent

        if not (dir_a / "config.json").is_file() or not (dir_b / "config.json").is_file():
            raise ModelCompatibilityError(
                "Activation diffing via Transformers requires `config.json` and tokenizer files in both model directories."
            )

        with self.resource_manager.managed_execution(cancellation_token) as telemetry:
            traces_a, tok_info_a, vocab_map_a = self._run_single_pretrained(
                dir_a, active_prompts, device=device, cancellation_token=cancellation_token
            )
            self.resource_manager.cleanup()
            traces_b, tok_info_b, vocab_map_b = self._run_single_pretrained(
                dir_b, active_prompts, device=device, cancellation_token=cancellation_token
            )
            self.resource_manager.cleanup()

        return self._build_report(
            traces_a=traces_a,
            traces_b=traces_b,
            model_a_id=meta_a.model_id,
            model_b_id=meta_b.model_id,
            revision_a=meta_a.revision,
            revision_b=meta_b.revision,
            tokenizer_a_info=tok_info_a,
            tokenizer_b_info=tok_info_b,
            decode_a=lambda tid: vocab_map_a.get(tid, f"<{tid}>"),
            decode_b=lambda tid: vocab_map_b.get(tid, f"<{tid}>"),
            device=device,
            telemetry=telemetry,
        )

    def _run_single_pretrained(
        self,
        model_dir: Path,
        prompts: list[str],
        *,
        device: str,
        cancellation_token: CancellationToken | None,
    ) -> tuple[list[CapturedPromptTrace], dict[str, Any], dict[int, str]]:
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=False,
            local_files_only=True,
        )
        tokenized: list[tuple[str, list[int], list[str]]] = []
        for p in prompts:
            enc = tokenizer(
                p,
                truncation=True,
                max_length=self.max_prompt_tokens,
                add_special_tokens=True,
            )
            ids = [int(x) for x in enc["input_ids"]]
            if not ids:
                fallback_id = int(tokenizer.bos_token_id or tokenizer.eos_token_id or 0)
                ids = [fallback_id]
            toks = [str(tokenizer.decode([tid])) for tid in ids]
            tokenized.append((p, ids, toks))

        torch_dtype = torch.bfloat16 if (device.startswith("cuda") and self.dtype == "bfloat16") else torch.float32

        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                trust_remote_code=False,
                local_files_only=True,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
        except Exception:
            model = AutoModel.from_pretrained(
                str(model_dir),
                trust_remote_code=False,
                local_files_only=True,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )

        nn.Module.to(model, device)
        try:
            traces = capture_module_traces(
                model,
                tokenized,
                device=device,
                seed=self.seed,
                cancellation_token=cancellation_token,
            )
            # Build lightweight decode map only for top-5 logit tokens across traces
            needed_ids: set[int] = set()
            for tr in traces:
                if tr.final_token_logits is not None and tr.final_token_logits.numel() > 0:
                    _, top_idx = torch.topk(tr.final_token_logits, k=min(10, tr.final_token_logits.numel()))
                    needed_ids.update(int(i) for i in top_idx.tolist())
            vocab_decode = {tid: str(tokenizer.decode([tid])) for tid in needed_ids}
        finally:
            del model
            self.resource_manager.cleanup()

        tok_info = {
            "tokenizer_class": type(tokenizer).__name__,
            "vocab_size": int(len(tokenizer)),
            "name_or_path": str(model_dir.name),
        }
        return traces, tok_info, vocab_decode

    def _build_report(
        self,
        *,
        traces_a: list[CapturedPromptTrace],
        traces_b: list[CapturedPromptTrace],
        model_a_id: str,
        model_b_id: str,
        revision_a: str | None,
        revision_b: str | None,
        tokenizer_a_info: dict[str, Any],
        tokenizer_b_info: dict[str, Any],
        decode_a,
        decode_b,
        device: str,
        telemetry: dict[str, Any],
    ) -> ActivationDiffReport:
        prompt_diffs: list[PromptActivationDiff] = []
        cross_notes: list[str] = []

        hidden_cos_accum: dict[int, list[float]] = {}
        hidden_cka_accum: dict[int, list[float]] = {}
        mlp_cos_accum: dict[int, list[float]] = {}
        attn_cos_accum: dict[int, list[float]] = {}
        js_vals: list[float] = []

        pooled_final_a: list[torch.Tensor] = []
        pooled_final_b: list[torch.Tensor] = []

        for ta, tb in zip(traces_a, traces_b, strict=True):
            p_limits: list[str] = []
            toks_identical = ta.token_ids == tb.token_ids
            if not toks_identical:
                msg = (
                    f"Token sequences differ ({len(ta.token_ids)} vs {len(tb.token_ids)} tokens); "
                    "direct elementwise cosine requires equal sequence length and identical hidden dimension."
                )
                p_limits.append(msg)
                if msg not in cross_notes:
                    cross_notes.append(msg)

            shared_hidden = sorted(set(ta.hidden_states.keys()) & set(tb.hidden_states.keys()))
            if set(ta.hidden_states.keys()) != set(tb.hidden_states.keys()):
                note = (
                    f"Layer depth differs ({len(ta.hidden_states)} vs {len(tb.hidden_states)}); "
                    f"compared {len(shared_hidden)} shared layer indices."
                )
                p_limits.append(note)
                if note not in cross_notes:
                    cross_notes.append(note)

            h_comps: list[ActivationLayerComparison] = []
            for lidx in shared_hidden:
                comp = compare_activation_tensors(
                    layer_name=f"layer.{lidx}.hidden_state",
                    component_type="hidden_state",
                    layer_index=lidx,
                    act_a=ta.hidden_states[lidx],
                    act_b=tb.hidden_states[lidx],
                )
                h_comps.append(comp)
                if comp.cosine_similarity is not None:
                    hidden_cos_accum.setdefault(lidx, []).append(comp.cosine_similarity)
                if comp.linear_cka is not None:
                    hidden_cka_accum.setdefault(lidx, []).append(comp.linear_cka)

            a_comps: list[ActivationLayerComparison] = []
            for lidx in sorted(set(ta.attention_outputs.keys()) & set(tb.attention_outputs.keys())):
                comp = compare_activation_tensors(
                    layer_name=f"layer.{lidx}.attention_output",
                    component_type="attention_output",
                    layer_index=lidx,
                    act_a=ta.attention_outputs[lidx],
                    act_b=tb.attention_outputs[lidx],
                )
                a_comps.append(comp)
                if comp.cosine_similarity is not None:
                    attn_cos_accum.setdefault(lidx, []).append(comp.cosine_similarity)

            m_comps: list[ActivationLayerComparison] = []
            for lidx in sorted(set(ta.mlp_intermediates.keys()) & set(tb.mlp_intermediates.keys())):
                comp = compare_activation_tensors(
                    layer_name=f"layer.{lidx}.mlp",
                    component_type="mlp_intermediate",
                    layer_index=lidx,
                    act_a=ta.mlp_intermediates[lidx],
                    act_b=tb.mlp_intermediates[lidx],
                )
                m_comps.append(comp)
                if comp.cosine_similarity is not None:
                    mlp_cos_accum.setdefault(lidx, []).append(comp.cosine_similarity)

            # Collect last-token final-layer hidden state for cross-prompt pooled CKA (works even cross-architecture!)
            if ta.hidden_states and tb.hidden_states:
                last_la = max(ta.hidden_states.keys())
                last_lb = max(tb.hidden_states.keys())
                ha_last = ta.hidden_states[last_la].squeeze(0)[-1]
                hb_last = tb.hidden_states[last_lb].squeeze(0)[-1]
                pooled_final_a.append(ha_last)
                pooled_final_b.append(hb_last)

            kl_div, js_div, top_a, top_b = _compute_logit_divergences(
                ta.final_token_logits, tb.final_token_logits, decode_a, decode_b
            )
            if js_div is not None and not math.isnan(js_div):
                js_vals.append(js_div)

            prompt_diffs.append(
                PromptActivationDiff(
                    prompt=ta.prompt,
                    token_ids_a=ta.token_ids,
                    token_ids_b=tb.token_ids,
                    tokens_a=ta.tokens,
                    tokens_b=tb.tokens,
                    tokens_identical=toks_identical,
                    next_token_kl_divergence=kl_div,
                    next_token_js_divergence=js_div,
                    top_tokens_a=top_a,
                    top_tokens_b=top_b,
                    hidden_state_comparisons=h_comps,
                    attention_comparisons=a_comps,
                    mlp_comparisons=m_comps,
                    limitations=p_limits,
                )
            )

        pooled_cka: float | None = None
        if len(pooled_final_a) >= 2 and len(pooled_final_b) >= 2:
            mat_pa = torch.stack(pooled_final_a, dim=0)
            mat_pb = torch.stack(pooled_final_b, dim=0)
            pooled_cka = compute_linear_cka(mat_pa, mat_pb)

        all_h_layers = sorted(set(hidden_cos_accum.keys()) | set(hidden_cka_accum.keys()))
        mean_h_cos = {
            idx: (round(sum(hidden_cos_accum[idx]) / len(hidden_cos_accum[idx]), 8) if idx in hidden_cos_accum else None)
            for idx in all_h_layers
        }
        mean_h_cka = {
            idx: (round(sum(hidden_cka_accum[idx]) / len(hidden_cka_accum[idx]), 8) if idx in hidden_cka_accum else None)
            for idx in all_h_layers
        }
        mean_m_cos: dict[int, float | None] = {
            idx: round(sum(vals) / len(vals), 8) for idx, vals in sorted(mlp_cos_accum.items()) if vals
        }
        mean_a_cos: dict[int, float | None] = {
            idx: round(sum(vals) / len(vals), 8) for idx, vals in sorted(attn_cos_accum.items()) if vals
        }

        return ActivationDiffReport(
            epistemic_status=EpistemicStatus.OBSERVED,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            revision_a=revision_a,
            revision_b=revision_b,
            tokenizer_a_info=tokenizer_a_info,
            tokenizer_b_info=tokenizer_b_info,
            inference_parameters={
                "seed": self.seed,
                "max_prompt_tokens": self.max_prompt_tokens,
                "dtype": self.dtype,
                "device": device,
                "do_sample": False,
                "trust_remote_code": False,
            },
            hardware_telemetry=telemetry,
            software_versions={
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "safetensors": safetensors.__version__,
            },
            prompt_count=len(prompt_diffs),
            mean_hidden_cosine_by_layer=mean_h_cos,
            mean_hidden_cka_by_layer=mean_h_cka,
            mean_mlp_cosine_by_layer=mean_m_cos,
            mean_attn_cosine_by_layer=mean_a_cos,
            mean_final_token_js_divergence=round(sum(js_vals) / len(js_vals), 8) if js_vals else None,
            cross_prompt_pooled_cka_final_layer=pooled_cka,
            prompt_diffs=prompt_diffs,
            cross_architecture_notes=cross_notes,
        )
