"""Strict Pydantic v2 API request and response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.orchestrator import VALID_MODULES


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class InspectModelRequest(StrictModel):
    source: str = Field(..., min_length=1, max_length=512)
    local_files_only: bool = False


class ValidateCompatibilityRequest(StrictModel):
    model_a: str = Field(..., min_length=1, max_length=512)
    model_b: str = Field(..., min_length=1, max_length=512)
    local_files_only: bool = False


class TolerancesInput(StrictModel):
    atol: float = Field(default=1e-6, ge=0.0, le=10.0)
    rtol: float = Field(default=1e-5, ge=0.0, le=10.0)
    zero_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class ThresholdsInput(StrictModel):
    policy_name: str = Field(default="production-default-v1", min_length=1, max_length=80)
    forbid_nan_or_inf: bool = True
    forbid_missing_tensors: bool = True
    forbid_shape_mismatches: bool = True
    max_overall_pass_rate_drop: float = Field(default=0.05, ge=0.0, le=1.0)
    critical_overall_pass_rate_drop: float = Field(default=0.15, ge=0.0, le=1.0)
    max_overall_score_drop: float = Field(default=0.05, ge=0.0, le=1.0)
    max_category_score_drop: float = Field(default=0.10, ge=0.0, le=1.0)
    critical_category_score_drop: float = Field(default=0.25, ge=0.0, le=1.0)
    zero_tolerance_refusal_regression: bool = True
    min_final_hidden_cka: float = Field(default=0.60, ge=0.0, le=1.0)


class RunComparisonRequest(StrictModel):
    model_a: str = Field(..., min_length=1, max_length=512)
    model_b: str = Field(..., min_length=1, max_length=512)
    modules: list[str] = Field(
        default_factory=lambda: ["structural", "tensor", "neuron", "regression"],
        min_length=1,
        max_length=10,
    )
    experiment_type: Literal[
        "REAL MODEL EXPERIMENT",
        "SYNTHETIC FIXTURE EXPERIMENT",
    ] = "SYNTHETIC FIXTURE EXPERIMENT"
    tolerances: TolerancesInput = Field(default_factory=TolerancesInput)
    thresholds: ThresholdsInput = Field(default_factory=ThresholdsInput)
    preferred_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    chunk_elements: int = Field(default=1_048_576, ge=4096, le=16_777_216)
    activation_prompts: list[str] | None = Field(default=None, max_length=16)
    behavioral_categories: list[str] | None = Field(default=None, max_length=8)
    behavioral_max_cases: int | None = Field(default=None, ge=1, le=200)
    max_new_tokens: int = Field(default=96, ge=8, le=512)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    prompt_mode: Literal["chat_if_available", "identical_raw", "qa_framed"] = "chat_if_available"
    local_files_only: bool = False
    baseline_id: str | None = Field(default=None, max_length=120)
    async_job: bool = False

    @field_validator("modules")
    @classmethod
    def _validate_modules(cls, values: list[str]) -> list[str]:
        invalid = [v for v in values if v not in VALID_MODULES]
        if invalid:
            raise ValueError(f"Unsupported module(s): {invalid}. Allowed: {list(VALID_MODULES)}")
        return list(dict.fromkeys(values))

    @field_validator("activation_prompts")
    @classmethod
    def _validate_prompts(cls, prompts: list[str] | None) -> list[str] | None:
        if prompts is None:
            return None
        cleaned = [p for p in prompts if p]
        for p in cleaned:
            if len(p) > 4096:
                raise ValueError("Individual activation prompt exceeds 4096 characters.")
        return cleaned or None


class CreateBaselineRequest(StrictModel):
    comparison_id: str = Field(..., min_length=3, max_length=80)
    baseline_id: str = Field(..., min_length=2, max_length=80, pattern=r"^[A-Za-z0-9._-]+$")
    side: Literal["a", "b"] = "a"


class RuntimeSettingsUpdate(StrictModel):
    preferred_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    atol: float = Field(default=1e-6, ge=0.0, le=10.0)
    rtol: float = Field(default=1e-5, ge=0.0, le=10.0)
    zero_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    chunk_elements: int = Field(default=1_048_576, ge=4096, le=16_777_216)
    allow_pytorch_weights_only: bool = True
    max_new_tokens: int = Field(default=96, ge=8, le=512)
