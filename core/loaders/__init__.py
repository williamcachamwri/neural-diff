"""Unified model loader exports (Module A)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.loaders.base import CheckpointReader, LoadedModel

if TYPE_CHECKING:
    from core.loaders.model_loader import ModelLoader, load_model

__all__ = ["CheckpointReader", "LoadedModel", "ModelLoader", "load_model"]


def __getattr__(name: str) -> Any:
    if name in ("ModelLoader", "load_model"):
        from core.loaders.model_loader import ModelLoader, load_model

        return ModelLoader if name == "ModelLoader" else load_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
