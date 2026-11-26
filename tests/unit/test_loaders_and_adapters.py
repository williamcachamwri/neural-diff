"""Unit tests for Safetensors multi-file, PyTorch sharded, and Hugging Face directory adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from adapters.huggingface.loader import (
    detect_snapshot_revision,
    open_directory_checkpoint,
    resolve_hf_hub_to_local_dir,
)
from adapters.pytorch.loader import SafePyTorchReader
from adapters.safetensors.loader import MultiSafetensorsReader, SingleSafetensorsReader
from core.exceptions import (
    MalformedCheckpointError,
    ModelNotFoundError,
    SecurityValidationError,
    UnsupportedFormatError,
)
from core.loaders.model_loader import ModelLoader
from core.security import SecurityPolicy


@pytest.mark.unit
def test_multi_file_safetensors_and_wide_row_chunking(tmp_path: Path) -> None:
    policy = SecurityPolicy(allowed_roots=[tmp_path])
    f1 = tmp_path / "part1.safetensors"
    f2 = tmp_path / "part2.safetensors"

    # Wide row `[2, 512]` streamed with tiny `chunk_elements=128` exercises row_elems > step_elems branch
    save_file({"wide.weight": torch.ones((2, 512))}, str(f1))
    save_file({"bias.weight": torch.zeros((16,))}, str(f2))

    single = SingleSafetensorsReader(f1, policy=policy)
    chunks = list(single.iter_tensor_chunks("wide.weight", chunk_elements=128))
    assert sum(c.numel() for c in chunks) == 1024

    multi = MultiSafetensorsReader.from_shard_files([f1, f2], policy=policy)
    assert multi.shard_count == 2
    assert set(multi.descriptors().keys()) == {"wide.weight", "bias.weight"}

    # Duplicate tensor across shards must raise MalformedCheckpointError
    f_dup = tmp_path / "part3.safetensors"
    save_file({"wide.weight": torch.ones((2, 512))}, str(f_dup))
    with pytest.raises(MalformedCheckpointError):
        MultiSafetensorsReader.from_shard_files([f1, f_dup], policy=policy)


@pytest.mark.unit
def test_pytorch_sharded_index_and_disabled_policy(tmp_path: Path) -> None:
    s1 = tmp_path / "pytorch_model-00001-of-00002.bin"
    s2 = tmp_path / "pytorch_model-00002-of-00002.bin"
    torch.save({"a.weight": torch.randn(4, 4)}, str(s1))
    torch.save({"b.weight": torch.randn(4, 4)}, str(s2))

    idx = tmp_path / "pytorch_model.bin.index.json"
    idx.write_text(
        json.dumps(
            {
                "weight_map": {
                    "a.weight": s1.name,
                    "b.weight": s2.name,
                }
            }
        ),
        encoding="utf-8",
    )

    policy_ok = SecurityPolicy(allowed_roots=[tmp_path], allow_pytorch_weights_only=True)
    with ModelLoader(policy=policy_ok).load(idx) as loaded:
        assert loaded.metadata.shard_count == 2
        assert loaded.get_tensor("a.weight").shape == (4, 4)

    # Disabling PyTorch pickle policy must reject loading `.bin`/`.pt`
    policy_strict = SecurityPolicy(allowed_roots=[tmp_path], allow_pytorch_weights_only=False)
    with pytest.raises(SecurityValidationError):
        SafePyTorchReader.from_single_file(s1, policy=policy_strict)


@pytest.mark.unit
def test_hf_directory_untrusted_py_ignore_and_snapshot_sha(tmp_path: Path) -> None:
    policy = SecurityPolicy(allowed_roots=[tmp_path])

    # Simulated HF cache snapshot path `snapshots/<40-hex>`
    fake_sha = "060db6499f32faf8b98477b0a26969ef7d8b9987"
    snap_dir = tmp_path / "snapshots" / fake_sha
    snap_dir.mkdir(parents=True)
    assert detect_snapshot_revision(snap_dir) == fake_sha

    # Untrusted custom modeling_evil.py inside directory must NEVER be executed and must be noted
    (snap_dir / "modeling_custom.py").write_text("raise RuntimeError('MUST NEVER EXECUTE')\n", encoding="utf-8")
    save_file({"w": torch.ones((4, 4))}, str(snap_dir / "custom_shard.safetensors"))

    reader, _fmt, shard_cnt, notes = open_directory_checkpoint(snap_dir, policy=policy)
    assert shard_cnt == 1
    assert any("Ignored 1 custom Python file" in n for n in notes)
    reader.close()

    # Empty directory raises MalformedCheckpointError
    empty_dir = tmp_path / "empty_model"
    empty_dir.mkdir()
    with pytest.raises(MalformedCheckpointError):
        open_directory_checkpoint(empty_dir, policy=policy)

    # Unsupported extension raises UnsupportedFormatError
    txt = tmp_path / "weights.onnx"
    txt.write_bytes(b"onnx")
    with pytest.raises(UnsupportedFormatError):
        ModelLoader(policy=policy).load(txt)

    # Invalid HF repo ID raises ModelNotFoundError
    with pytest.raises(ModelNotFoundError):
        resolve_hf_hub_to_local_dir("../invalid_repo_id", local_files_only=True)
