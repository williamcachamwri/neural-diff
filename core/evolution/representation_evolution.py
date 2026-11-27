"""Representation Evolution Engine (Section 7).

Integrates forward-hook hidden-state capture, Centered Linear CKA, hidden-state cosine
similarity, activation & logit distribution drift, Logit Lens (when architecture-compatible),
and optional cross-checkpoint residual-stream Activation Patching (`CAUSAL SUPPORT` on generic
diagnostic transition prompts vs `CAUSALLY VERIFIED` strictly on case-targeted failed prompts).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from core.activation_analysis.comparator import (
    DEFAULT_DIAGNOSTIC_PROMPTS,
    CapturedPromptTrace,
    capture_module_traces,
)
from core.activation_analysis.logit_lens import run_logit_lens
from core.activation_analysis.metrics import compute_linear_cka
from core.evolution.dag import EvolutionDAG
from core.evolution.types import CheckpointValidationStatus, RegisteredCheckpoint
from core.resource_manager import CancellationToken, ResourceManager
from core.types import EpistemicStatus


def _js_divergence_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float | None:
    if logits_a.shape != logits_b.shape or logits_a.numel() == 0:
        return None
    pa = torch.softmax(logits_a.reshape(-1).double(), dim=-1).clamp_min(1e-12)
    pb = torch.softmax(logits_b.reshape(-1).double(), dim=-1).clamp_min(1e-12)
    m = 0.5 * (pa + pb)
    kl_am = float(torch.sum(pa * (torch.log(pa) - torch.log(m))).item())
    kl_bm = float(torch.sum(pb * (torch.log(pb) - torch.log(m))).item())
    return round(max(0.0, 0.5 * (kl_am + kl_bm)), 8)


def _logit_entropy(logits: torch.Tensor) -> float:
    p = torch.softmax(logits.reshape(-1).double(), dim=-1).clamp_min(1e-12)
    return round(float((-torch.sum(p * torch.log(p))).item()), 6)


def _find_decoder_layers_and_norms(
    model: nn.Module,
) -> tuple[nn.ModuleList | None, nn.Module | None, nn.Module | None]:
    """Locate decoder block `ModuleList`, `final_norm`, and `lm_head` across Llama/Qwen/SmolLM/GPT2."""
    lm_head = getattr(model, "lm_head", None)
    inner_model = getattr(model, "model", None)
    if inner_model is not None:
        layers = getattr(inner_model, "layers", None)
        norm = getattr(inner_model, "norm", None)
        if isinstance(layers, nn.ModuleList):
            return layers, norm if isinstance(norm, nn.Module) else None, lm_head if isinstance(lm_head, nn.Module) else None

    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        h = getattr(transformer, "h", None)
        ln_f = getattr(transformer, "ln_f", None)
        if isinstance(h, nn.ModuleList):
            return h, ln_f if isinstance(ln_f, nn.Module) else None, lm_head if isinstance(lm_head, nn.Module) else None

    return None, None, None


def _deterministic_fallback_tokens(prompt: str, vocab_size: int, seq_len: int = 12) -> list[int]:
    """Generate deterministic token IDs in `[1, vocab_size - 1]` when a synthetic HF dir lacks tokenizer files."""
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    safe_vocab = max(4, int(vocab_size))
    return [1 + (digest[i % len(digest)] % (safe_vocab - 1)) for i in range(seq_len)]


@dataclass
class CheckpointTraceBundle:
    """Compact CPU-resident representation traces captured from one checkpoint."""

    checkpoint_id: str
    version_label: str
    architecture: str | None
    model_family: str | None
    hidden_size: int | None
    layer_count: int | None
    vocab_size: int | None
    tokenizer_signature: str
    traces: list[CapturedPromptTrace]
    logit_lens_summary: dict[str, Any] | None


@torch.no_grad()
def run_cross_checkpoint_layer_patching(
    clean_dir: Path,
    corrupted_dir: Path,
    input_ids: torch.Tensor,
    *,
    device: str = "cpu",
    prompt_text: str | None = None,
) -> dict[str, Any] | None:
    """Interventional cross-checkpoint patching: patch clean Model A block outputs into Model B one layer at a time.

    Labeled `CAUSAL SUPPORT` because it operates on a generic diagnostic prompt rather than
    a case-targeted failed behavioral rubric case (`case_targeted=False`).
    """
    from transformers import AutoModelForCausalLM

    model_a = AutoModelForCausalLM.from_pretrained(
        str(clean_dir),
        trust_remote_code=False,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    torch.nn.Module.to(model_a, device)
    model_a.eval()

    layers_a, _, _ = _find_decoder_layers_and_norms(model_a)
    if layers_a is None or len(layers_a) == 0:
        del model_a
        return None

    clean_acts: dict[int, torch.Tensor] = {}
    handles_a = []

    def _cap(idx: int):
        def _hook(_m: nn.Module, _inp: Any, out: Any) -> None:
            hs = out[0] if isinstance(out, tuple) else out
            clean_acts[idx] = hs.detach().clone()

        return _hook

    for i, blk in enumerate(layers_a):
        handles_a.append(blk.register_forward_hook(_cap(i)))

    try:
        logits_a = model_a(input_ids=input_ids.to(device)).logits[0, -1].detach().cpu().double()
    finally:
        for h in handles_a:
            h.remove()
        del model_a

    top2 = torch.topk(logits_a, k=min(2, logits_a.numel()))
    target_tok = int(top2.indices[0].item())
    foil_tok = int(top2.indices[1].item()) if top2.indices.numel() > 1 else 0
    clean_diff = float((logits_a[target_tok] - logits_a[foil_tok]).item())

    model_b = AutoModelForCausalLM.from_pretrained(
        str(corrupted_dir),
        trust_remote_code=False,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    torch.nn.Module.to(model_b, device)
    model_b.eval()

    layers_b, _, _ = _find_decoder_layers_and_norms(model_b)
    if layers_b is None or len(layers_b) != len(clean_acts):
        del model_b
        return None

    try:
        logits_b = model_b(input_ids=input_ids.to(device)).logits[0, -1].detach().cpu().double()
        corrupted_diff = float((logits_b[target_tok] - logits_b[foil_tok]).item())
        denom = clean_diff - corrupted_diff

        layer_effects: list[dict[str, Any]] = []
        for idx, blk_b in enumerate(layers_b):
            clean_tensor = clean_acts[idx].to(device)
            if clean_tensor.shape[-1] != logits_b.shape[0] and False:
                pass

            def _patch_hook(_m: nn.Module, _inp: Any, out: Any, ct: torch.Tensor = clean_tensor) -> Any:
                if isinstance(out, tuple):
                    if out[0].shape != ct.shape:
                        return out
                    return (ct, *out[1:])
                if out.shape != ct.shape:
                    return out
                return ct

            h_patch = blk_b.register_forward_hook(_patch_hook)
            try:
                patched_logits = (
                    model_b(input_ids=input_ids.to(device)).logits[0, -1].detach().cpu().double()
                )
            finally:
                h_patch.remove()

            p_diff = float((patched_logits[target_tok] - patched_logits[foil_tok]).item())
            recovery = 0.0 if denom <= 0.05 else (p_diff - corrupted_diff) / denom
            layer_effects.append(
                {
                    "layer_index": idx,
                    "patched_logit_diff": round(p_diff, 6),
                    "recovery_fraction": round(recovery, 6),
                }
            )
    finally:
        del model_b

    top_causal = sorted(
        layer_effects, key=lambda x: float(x["recovery_fraction"]), reverse=True
    )
    peak_rec = float(top_causal[0]["recovery_fraction"]) if top_causal else 0.0
    most_layer = int(top_causal[0]["layer_index"]) if top_causal else None
    non_top = top_causal[1:]
    min_control_entry = (
        min(non_top, key=lambda x: abs(float(x["recovery_fraction"]))) if non_top else None
    )
    min_control_abs = (
        abs(float(min_control_entry["recovery_fraction"])) if min_control_entry is not None else 0.0
    )
    prompt_raw = (
        prompt_text.encode("utf-8")
        if prompt_text is not None
        else ",".join(str(int(x)) for x in input_ids.reshape(-1).tolist()).encode("utf-8")
    )
    prompt_fp = hashlib.sha256(prompt_raw).hexdigest()
    target_foil_ok = bool(
        target_tok != foil_tok and denom > 0.05 and peak_rec >= 0.20
    )
    neg_ctrl_ok = bool(
        min_control_entry is not None and min_control_abs < 0.15 and min_control_abs < peak_rec
    )
    return {
        "epistemic_status": (
            EpistemicStatus.CAUSAL_SUPPORT.value
            if peak_rec >= 0.20 and denom > 0.05
            else EpistemicStatus.OBSERVED.value
        ),
        "case_targeted": False,
        "target_vs_foil_verified": target_foil_ok,
        "negative_control_verified": neg_ctrl_ok,
        "verification_scope": "diagnostic_prompt_transition_only",
        "prompt_fingerprint": prompt_fp,
        "failed_case_id": None,
        "layer_index": most_layer,
        "intervention_type": "residual_stream_layer_patch",
        "target_token": target_tok,
        "foil_token": foil_tok,
        "negative_control_result": {
            "control_layer_index": (
                int(min_control_entry["layer_index"]) if min_control_entry is not None else None
            ),
            "control_recovery_fraction": round(min_control_abs, 6),
        },
        "recovery_measurement": round(peak_rec, 6),
        "prompt_level_causal_verification": peak_rec >= 0.20 and denom > 0.05,
        "intervention_target": "diagnostic_prompt",
        "scope": "cross-checkpoint single-layer residual stream intervention on generic diagnostic prompt (supports transition hypothesis; does not causally verify unpatched behavioral rubric cases)",
        "target_token_id": target_tok,
        "foil_token_id": foil_tok,
        "clean_logit_diff": round(clean_diff, 6),
        "corrupted_logit_diff": round(corrupted_diff, 6),
        "most_causal_layer": most_layer,
        "peak_recovery_fraction": peak_rec,
        "min_control_layer_abs_recovery": round(min_control_abs, 6),
        "layer_effects": layer_effects,
    }


class RepresentationEvolutionEngine:
    """Computes multi-generation CKA, hidden-state cosine, drift, Logit Lens, and causal patching."""

    def __init__(
        self,
        *,
        resource_manager: ResourceManager | None = None,
        prompts: list[str] | None = None,
        seed: int = 42,
        enable_causal_patching: bool = True,
    ) -> None:
        self.resource_manager = resource_manager or ResourceManager()
        self.prompts = list(prompts or DEFAULT_DIAGNOSTIC_PROMPTS[:3])
        self.seed = seed
        self.enable_causal_patching = enable_causal_patching

    def _capture_checkpoint(
        self,
        ckpt: RegisteredCheckpoint,
        *,
        device: str,
        cancellation_token: CancellationToken | None = None,
    ) -> CheckpointTraceBundle:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if ckpt.validation_status == CheckpointValidationStatus.ADAPTER_ONLY:
            raise ValueError(f"Checkpoint '{ckpt.version_label}' contains LoRA adapter weights only.")

        resolved = Path(ckpt.resolved_path)
        model_dir = resolved if resolved.is_dir() else resolved.parent
        if not (model_dir / "config.json").is_file():
            raise ValueError(f"Checkpoint '{ckpt.version_label}' lacks `config.json` for forward-pass hooks.")

        has_tok = (model_dir / "tokenizer.json").is_file() or (
            model_dir / "tokenizer_config.json"
        ).is_file()
        tokenized_prompts: list[tuple[str, list[int], list[str]]] = []
        if has_tok:
            tok = AutoTokenizer.from_pretrained(
                str(model_dir), trust_remote_code=False, local_files_only=True
            )
            tok_sig = f"{tok.__class__.__name__}:{len(tok)}"
            for p in self.prompts:
                ids = list(tok.encode(p, add_special_tokens=True))[:64]
                if not ids:
                    ids = [int(tok.eos_token_id or 0)]
                strs = [str(tok.decode([tid])) for tid in ids]
                tokenized_prompts.append((p, ids, strs))
        else:
            vocab_sz = ckpt.vocab_size or 128
            tok_sig = f"synthetic_deterministic_ids:{vocab_sz}"
            for p in self.prompts:
                ids = _deterministic_fallback_tokens(p, vocab_sz, seq_len=12)
                tokenized_prompts.append((p, ids, [f"tok_{tid}" for tid in ids]))

        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=False,
            local_files_only=True,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        torch.nn.Module.to(model, device)
        model.eval()

        logit_lens_summary: dict[str, Any] | None = None
        try:
            traces = capture_module_traces(
                model,
                tokenized_prompts,
                device=device,
                seed=self.seed,
                cancellation_token=cancellation_token,
            )
            # Run Logit Lens on first prompt when final_norm and lm_head are architecture-compatible
            _, final_norm, lm_head = _find_decoder_layers_and_norms(model)
            if final_norm is not None and lm_head is not None and tokenized_prompts:
                first_ids = torch.tensor(
                    [tokenized_prompts[0][1]], dtype=torch.long, device=device
                )
                try:
                    lens_res = run_logit_lens(
                        model,
                        first_ids,
                        final_norm=final_norm,
                        lm_head=lm_head,
                        top_k=3,
                    )
                    logit_lens_summary = lens_res.to_dict()
                except Exception as exc:
                    _ = exc
        finally:
            del model
            self.resource_manager.cleanup()

        return CheckpointTraceBundle(
            checkpoint_id=ckpt.checkpoint_id,
            version_label=ckpt.version_label,
            architecture=ckpt.architecture,
            model_family=ckpt.model_family,
            hidden_size=ckpt.hidden_size,
            layer_count=ckpt.layer_count,
            vocab_size=ckpt.vocab_size,
            tokenizer_signature=tok_sig,
            traces=traces,
            logit_lens_summary=logit_lens_summary,
        )

    @staticmethod
    def _compare_trace_bundles(
        bundle_a: CheckpointTraceBundle,
        bundle_b: CheckpointTraceBundle,
    ) -> dict[str, Any]:
        compat_notes: list[str] = []
        compatible = True

        if bundle_a.layer_count != bundle_b.layer_count:
            compat_notes.append(
                f"Layer count mismatch ({bundle_a.layer_count} vs {bundle_b.layer_count}); comparing shared layer indices only."
            )
        if bundle_a.hidden_size != bundle_b.hidden_size:
            compat_notes.append(
                f"Hidden dimension mismatch ({bundle_a.hidden_size} vs {bundle_b.hidden_size}); cosine similarity disabled, Linear CKA valid only on token-aligned prompts."
            )

        shared_layers: set[int] = set()
        for ta, tb in zip(bundle_a.traces, bundle_b.traces, strict=False):
            shared_layers |= set(ta.hidden_states.keys()) & set(tb.hidden_states.keys())

        sorted_layers = sorted(shared_layers)
        cka_by_layer: dict[int, float | None] = {}
        cos_by_layer: dict[int, float | None] = {}
        act_norm_delta_by_layer: dict[int, float | None] = {}

        token_aligned_prompts = 0
        for ta, tb in zip(bundle_a.traces, bundle_b.traces, strict=False):
            if ta.token_ids == tb.token_ids:
                token_aligned_prompts += 1

        if token_aligned_prompts == 0:
            compatible = False
            compat_notes.append(
                "Incompatible tokenization across diagnostic prompts (token sequences differ in length/IDs); direct token-aligned representation comparison blocked."
            )
            return {
                "compatible": False,
                "compatibility_warnings": compat_notes,
                "layer_indices": sorted_layers,
                "cka_by_layer": {str(lid): None for lid in sorted_layers},
                "cosine_by_layer": {str(lid): None for lid in sorted_layers},
                "activation_norm_delta_by_layer": {str(lid): None for lid in sorted_layers},
                "mean_cka": None,
                "mean_drift": None,
                "mean_hidden_cosine": None,
                "mean_logit_js_divergence": None,
                "logit_entropy_delta": None,
            }

        for l_idx in sorted_layers:
            pooled_a: list[torch.Tensor] = []
            pooled_b: list[torch.Tensor] = []
            cos_vals: list[float] = []
            norm_deltas: list[float] = []

            for ta, tb in zip(bundle_a.traces, bundle_b.traces, strict=False):
                if ta.token_ids != tb.token_ids:
                    continue
                ha = ta.hidden_states.get(l_idx)
                hb = tb.hidden_states.get(l_idx)
                if ha is None or hb is None:
                    continue
                fa = ha.reshape(-1, ha.shape[-1])
                fb = hb.reshape(-1, hb.shape[-1])
                if fa.shape[0] != fb.shape[0]:
                    continue
                pooled_a.append(fa)
                pooled_b.append(fb)

                na = float(torch.linalg.vector_norm(fa.double()).item()) / max(1, fa.shape[0])
                nb = float(torch.linalg.vector_norm(fb.double()).item()) / max(1, fb.shape[0])
                norm_deltas.append(nb - na)

                if fa.shape == fb.shape:
                    va = fa.reshape(-1).double()
                    vb = fb.reshape(-1).double()
                    denom = float((torch.linalg.vector_norm(va) * torch.linalg.vector_norm(vb)).item())
                    if denom > 1e-12:
                        cos_vals.append(max(-1.0, min(1.0, float(torch.dot(va, vb).item()) / denom)))
                    elif float(torch.linalg.vector_norm(va - vb).item()) == 0.0:
                        cos_vals.append(1.0)

            if pooled_a and pooled_b:
                cat_a = torch.cat(pooled_a, dim=0)
                cat_b = torch.cat(pooled_b, dim=0)
                cka_val = compute_linear_cka(cat_a, cat_b)
                cka_by_layer[l_idx] = round(cka_val, 6) if cka_val is not None else None
            else:
                cka_by_layer[l_idx] = None

            cos_by_layer[l_idx] = round(sum(cos_vals) / len(cos_vals), 6) if cos_vals else None
            act_norm_delta_by_layer[l_idx] = (
                round(sum(norm_deltas) / len(norm_deltas), 6) if norm_deltas else None
            )

        js_vals: list[float] = []
        ent_deltas: list[float] = []
        for ta, tb in zip(bundle_a.traces, bundle_b.traces, strict=False):
            if ta.final_token_logits is not None and tb.final_token_logits is not None:
                js = _js_divergence_from_logits(ta.final_token_logits, tb.final_token_logits)
                if js is not None:
                    js_vals.append(js)
                    ent_deltas.append(
                        _logit_entropy(tb.final_token_logits) - _logit_entropy(ta.final_token_logits)
                    )

        valid_ckas = [v for v in cka_by_layer.values() if v is not None and math.isfinite(v)]
        valid_cos = [v for v in cos_by_layer.values() if v is not None and math.isfinite(v)]
        mean_cka = round(sum(valid_ckas) / len(valid_ckas), 6) if valid_ckas else None
        mean_drift = round(1.0 - mean_cka, 6) if mean_cka is not None else None
        mean_cos = round(sum(valid_cos) / len(valid_cos), 6) if valid_cos else None

        return {
            "compatible": compatible,
            "compatibility_warnings": compat_notes,
            "layer_indices": sorted_layers,
            "cka_by_layer": {str(lid): cka_by_layer[lid] for lid in sorted_layers},
            "cosine_by_layer": {str(lid): cos_by_layer[lid] for lid in sorted_layers},
            "activation_norm_delta_by_layer": {
                str(lid): act_norm_delta_by_layer[lid] for lid in sorted_layers
            },
            "mean_cka": mean_cka,
            "mean_drift": mean_drift,
            "mean_hidden_cosine": mean_cos,
            "mean_logit_js_divergence": (
                round(sum(js_vals) / len(js_vals), 6) if js_vals else None
            ),
            "logit_entropy_delta": (
                round(sum(ent_deltas) / len(ent_deltas), 6) if ent_deltas else None
            ),
        }

    def run(
        self,
        dag: EvolutionDAG,
        *,
        base_checkpoint_id: str | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        """Capture traces sequentially across checkpoints and build CKA/Cosine evolution matrices."""
        device = self.resource_manager.resolve_device()
        topo = dag.topological_order()
        bundles: dict[str, CheckpointTraceBundle] = {}
        warnings: list[str] = []

        for cid in topo:
            if cancellation_token is not None:
                cancellation_token.check()
            ckpt = dag.checkpoints_by_id[cid]
            try:
                bundles[cid] = self._capture_checkpoint(
                    ckpt, device=device, cancellation_token=cancellation_token
                )
            except Exception as exc:
                warnings.append(
                    f"Representation capture skipped for '{ckpt.version_label}': {exc}"
                )

        roots = dag.roots()
        eff_base = base_checkpoint_id or (roots[0] if roots else None)

        # Build Representation Evolution Matrix (Checkpoints × Layers relative to Base Model)
        matrix_rows: list[dict[str, Any]] = []
        all_layers: set[int] = set()
        for b in bundles.values():
            for tr in b.traces:
                all_layers.update(tr.hidden_states.keys())
        sorted_layers = sorted(all_layers)

        if eff_base and eff_base in bundles:
            base_bundle = bundles[eff_base]
            for cid in topo:
                if cid not in bundles:
                    continue
                curr_bundle = bundles[cid]
                if cid == eff_base:
                    matrix_rows.append(
                        {
                            "checkpoint_id": cid,
                            "version_label": curr_bundle.version_label,
                            "transformation_type": dag.checkpoints_by_id[cid].transformation_type.value,
                            "compatible_with_base": True,
                            "compatibility_warnings": [],
                            "layer_cka": [1.0 for _ in sorted_layers],
                            "layer_cosine": [1.0 for _ in sorted_layers],
                            "mean_cka_to_base": 1.0,
                            "representation_drift": 0.0,
                            "mean_logit_js_to_base": 0.0,
                        }
                    )
                else:
                    cmp_res = self._compare_trace_bundles(base_bundle, curr_bundle)
                    matrix_rows.append(
                        {
                            "checkpoint_id": cid,
                            "version_label": curr_bundle.version_label,
                            "transformation_type": dag.checkpoints_by_id[cid].transformation_type.value,
                            "compatible_with_base": cmp_res["compatible"],
                            "compatibility_warnings": cmp_res["compatibility_warnings"],
                            "layer_cka": [
                                cmp_res["cka_by_layer"].get(str(lid)) for lid in sorted_layers
                            ],
                            "layer_cosine": [
                                cmp_res["cosine_by_layer"].get(str(lid)) for lid in sorted_layers
                            ],
                            "mean_cka_to_base": cmp_res["mean_cka"],
                            "representation_drift": cmp_res["mean_drift"],
                            "mean_logit_js_to_base": cmp_res["mean_logit_js_divergence"],
                        }
                    )

        # Consecutive transitions (`parent -> child`) + optional Causal Activation Patching on largest drift edge
        consecutive_diffs: list[dict[str, Any]] = []
        layer_drift_counts: dict[int, int] = {}

        for rel in dag.relationships:
            pa_id = rel.parent_checkpoint_id
            ch_id = rel.child_checkpoint_id
            if pa_id not in bundles or ch_id not in bundles:
                continue
            cmp_edge = self._compare_trace_bundles(bundles[pa_id], bundles[ch_id])
            for lid in sorted_layers:
                cval = cmp_edge["cka_by_layer"].get(str(lid))
                if cval is not None and cval < 0.999:
                    layer_drift_counts[lid] = layer_drift_counts.get(lid, 0) + 1

            patching_payload: dict[str, Any] | None = None
            pa_ckpt = dag.checkpoints_by_id[pa_id]
            ch_ckpt = dag.checkpoints_by_id[ch_id]
            if (
                self.enable_causal_patching
                and cmp_edge["compatible"]
                and pa_ckpt.hidden_size == ch_ckpt.hidden_size
                and pa_ckpt.layer_count == ch_ckpt.layer_count
                and pa_ckpt.vocab_size == ch_ckpt.vocab_size
                and bundles[pa_id].traces
            ):
                p_dir = (
                    Path(pa_ckpt.resolved_path)
                    if Path(pa_ckpt.resolved_path).is_dir()
                    else Path(pa_ckpt.resolved_path).parent
                )
                c_dir = (
                    Path(ch_ckpt.resolved_path)
                    if Path(ch_ckpt.resolved_path).is_dir()
                    else Path(ch_ckpt.resolved_path).parent
                )
                first_tok_ids = torch.tensor(
                    [bundles[pa_id].traces[0].token_ids], dtype=torch.long
                )
                try:
                    patching_payload = run_cross_checkpoint_layer_patching(
                        p_dir,
                        c_dir,
                        first_tok_ids,
                        device=device,
                        prompt_text=bundles[pa_id].traces[0].prompt,
                    )
                except Exception as exc:
                    _ = exc

            consecutive_diffs.append(
                {
                    "source_checkpoint_id": pa_id,
                    "source_label": pa_ckpt.version_label,
                    "target_checkpoint_id": ch_id,
                    "target_label": ch_ckpt.version_label,
                    "transformation_type": rel.transformation_type.value,
                    **cmp_edge,
                    "causal_patching": patching_payload,
                }
            )

        n_edges = len(consecutive_diffs)
        consistent_drift_layers = [
            lid
            for lid, cnt in sorted(layer_drift_counts.items())
            if n_edges >= 2 and (cnt / n_edges) >= 0.75
        ]

        logit_lens_by_checkpoint = {
            cid: bundles[cid].logit_lens_summary
            for cid in topo
            if cid in bundles and bundles[cid].logit_lens_summary is not None
        }

        return (
            {
                "epistemic_status": EpistemicStatus.OBSERVED.value,
                "prompt_count": len(self.prompts),
                "prompts": self.prompts,
                "layer_indices": sorted_layers,
                "evolution_matrix": {
                    "base_checkpoint_id": eff_base,
                    "layer_indices": sorted_layers,
                    "rows": matrix_rows,
                },
                "consecutive_transitions": consecutive_diffs,
                "consistent_drift_layers": consistent_drift_layers,
                "logit_lens_by_checkpoint": logit_lens_by_checkpoint,
            },
            warnings,
        )
