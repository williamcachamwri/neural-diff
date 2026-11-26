"""Base abstractions for lazy, memory-safe checkpoint reading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from types import TracebackType

import torch

from core.types import ModelMetadata, TensorDescriptor


class CheckpointReader(ABC):
    """Abstract lazy reader for checkpoint weight tensors."""

    @abstractmethod
    def descriptors(self) -> dict[str, TensorDescriptor]:
        """Return ordered mapping of tensor names to metadata descriptors."""

    @abstractmethod
    def get_tensor(self, name: str) -> torch.Tensor:
        """Materialize a single CPU tensor by name."""

    def iter_tensor_chunks(self, name: str, chunk_elements: int = 1_048_576) -> Iterator[torch.Tensor]:
        """Yield flattened 1D CPU tensor chunks containing at most `chunk_elements` values."""
        tensor = self.get_tensor(name).reshape(-1)
        total = tensor.numel()
        if total == 0:
            yield tensor
            return
        step = max(1, int(chunk_elements))
        for start in range(0, total, step):
            yield tensor[start : start + step]

    @abstractmethod
    def custom_metadata(self) -> dict[str, str]:
        """Return string-valued checkpoint metadata dictionary when present."""

    def close(self) -> None:  # noqa: B027
        """Release underlying file descriptors or memory-mapped handles."""


class LoadedModel:
    """Context-managed wrapper combining `ModelMetadata` and lazy `CheckpointReader`."""

    def __init__(self, metadata: ModelMetadata, reader: CheckpointReader) -> None:
        self.metadata = metadata
        self.reader = reader
        self._descriptors = reader.descriptors()

    @property
    def tensors(self) -> dict[str, TensorDescriptor]:
        return self._descriptors

    def tensor_names(self) -> list[str]:
        return list(self._descriptors.keys())

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.reader.get_tensor(name)

    def iter_tensor_chunks(self, name: str, chunk_elements: int = 1_048_576) -> Iterator[torch.Tensor]:
        return self.reader.iter_tensor_chunks(name, chunk_elements=chunk_elements)

    def close(self) -> None:
        self.reader.close()

    def __enter__(self) -> LoadedModel:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
