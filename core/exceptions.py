"""Structured domain and security exceptions for NEURAL DIFF."""

from __future__ import annotations

from typing import Any


class NeuralDiffError(Exception):
    """Base exception for all predictable NEURAL DIFF failures."""

    code: str = "NEURAL_DIFF_ERROR"
    http_status: int = 400

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        http_status: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if http_status is not None:
            self.http_status = http_status
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "http_status": self.http_status,
            "details": self.details,
        }


class SecurityValidationError(NeuralDiffError):
    """Raised on path traversal, disallowed path access, oversized headers, or unsafe pickle files."""

    code = "SECURITY_VALIDATION_ERROR"
    http_status = 403


class MalformedCheckpointError(NeuralDiffError):
    """Raised when a checkpoint file, shard index, or JSON config is corrupted or malformed."""

    code = "MALFORMED_CHECKPOINT_ERROR"
    http_status = 422


class UnsupportedFormatError(NeuralDiffError):
    """Raised when a model file format is unrecognized or unsupported."""

    code = "UNSUPPORTED_FORMAT_ERROR"
    http_status = 415


class ModelNotFoundError(NeuralDiffError):
    """Raised when a requested local path or Hugging Face model repository does not exist."""

    code = "MODEL_NOT_FOUND_ERROR"
    http_status = 404


class ModelAccessBlockedError(NeuralDiffError):
    """Raised when an upstream model is gated/private and credentials are unavailable."""

    code = "MODEL_ACCESS_BLOCKED"
    http_status = 403


class ModelCompatibilityError(NeuralDiffError):
    """Raised when an operation requires compatible architectures/shapes that models lack."""

    code = "MODEL_COMPATIBILITY_ERROR"
    http_status = 409


class ResourceExhaustedError(NeuralDiffError):
    """Raised when an operation exceeds configured memory, tensor size, or payload limits."""

    code = "RESOURCE_EXHAUSTED_ERROR"
    http_status = 503


class OperationCancelledError(NeuralDiffError):
    """Raised when an active comparison or benchmark is cancelled via CancellationToken."""

    code = "OPERATION_CANCELLED"
    http_status = 499
