"""Unit, DAG, Cycle Detection, Duplicate Hash, and SQLite Database tests for Neural Evolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.evolution.dag import EvolutionDAG
from core.evolution.registry import inspect_and_build_checkpoint_record
from core.evolution.store import EvolutionStore
from core.evolution.types import (
    CheckpointValidationStatus,
    TransformationType,
    normalize_transformation_type,
)
from core.exceptions import (
    CyclicDependencyError,
    EvolutionEntityNotFoundError,
    EvolutionGraphError,
)
from core.security import SecurityPolicy
from tests.fixtures.evolution_factory import create_known_evolution_checkpoints


@pytest.mark.unit
def test_all_14_transformation_types_normalized() -> None:
    expected = {
        "base_model": TransformationType.BASE_MODEL,
        "Fine-Tuning": TransformationType.FINE_TUNING,
        "instruction tuning": TransformationType.INSTRUCTION_TUNING,
        "sft": TransformationType.SFT,
        "DPO": TransformationType.DPO,
        "rlhf": TransformationType.RLHF,
        "lora": TransformationType.LORA,
        "QLoRA": TransformationType.QLORA,
        "model merging": TransformationType.MODEL_MERGING,
        "quantization": TransformationType.QUANTIZATION,
        "pruned": TransformationType.PRUNING,
        "distillation": TransformationType.DISTILLATION,
        "weight_modification": TransformationType.WEIGHT_MODIFICATION,
        "unrecognized_custom_stage": TransformationType.UNKNOWN_TRANSFORMATION,
    }
    for raw, target in expected.items():
        assert normalize_transformation_type(raw) == target
    assert len(list(TransformationType)) == 14


@pytest.mark.unit
def test_database_migrations_branched_dag_merging_and_cycle_detection(tmp_path: Path) -> None:
    ckpts = create_known_evolution_checkpoints(tmp_path)
    policy = SecurityPolicy(allowed_roots=[tmp_path])
    store = EvolutionStore(db_path=tmp_path / "evo_unit.db")

    # 1. Verify schema migrations recorded with SHA-256 checksums
    migs = store.list_migrations()
    assert len(migs) >= 1
    assert migs[0]["migration_id"] == "0001_evolution_core_tables"
    assert len(migs[0]["checksum_sha256"]) == 64

    proj = store.create_project(
        name="Branched & Merged Evolution",
        description="Tests branching, merging, duplicates, LoRA, and cycle rejection.",
        tags=["dag", "merge"],
    )

    # 2. Register Base -> SFT A -> DPO A, and Base -> SFT B, then Merge [DPO A, SFT B] -> Merged
    rec_base = store.add_checkpoint(
        inspect_and_build_checkpoint_record(
            project_id=proj.project_id,
            source=ckpts["a"],
            version_label="Base",
            transformation_type="Base Model",
            policy=policy,
            local_files_only=True,
        )
    )
    assert store.get_project(proj.project_id).base_checkpoint_id == rec_base.checkpoint_id

    rec_sft_a = store.add_checkpoint(
        inspect_and_build_checkpoint_record(
            project_id=proj.project_id,
            source=ckpts["b"],
            version_label="SFT-A",
            parent_checkpoint_ids=[rec_base.checkpoint_id],
            transformation_type="SFT",
            existing_project_checkpoints=store.list_checkpoints(proj.project_id),
            policy=policy,
            local_files_only=True,
        )
    )
    rec_dpo_a = store.add_checkpoint(
        inspect_and_build_checkpoint_record(
            project_id=proj.project_id,
            source=ckpts["c"],
            version_label="DPO-A",
            parent_checkpoint_ids=[rec_sft_a.checkpoint_id],
            transformation_type="DPO",
            existing_project_checkpoints=store.list_checkpoints(proj.project_id),
            policy=policy,
            local_files_only=True,
        )
    )
    rec_sft_b = store.add_checkpoint(
        inspect_and_build_checkpoint_record(
            project_id=proj.project_id,
            source=ckpts["sft_b"],
            version_label="SFT-B",
            parent_checkpoint_ids=[rec_base.checkpoint_id],
            transformation_type="SFT",
            existing_project_checkpoints=store.list_checkpoints(proj.project_id),
            policy=policy,
            local_files_only=True,
        )
    )
    rec_merged = store.add_checkpoint(
        inspect_and_build_checkpoint_record(
            project_id=proj.project_id,
            source=ckpts["merged"],
            version_label="Merged-AB",
            parent_checkpoint_ids=[rec_dpo_a.checkpoint_id, rec_sft_b.checkpoint_id],
            transformation_type="Model Merging",
            existing_project_checkpoints=store.list_checkpoints(proj.project_id),
            policy=policy,
            local_files_only=True,
        ),
        merge_weights={rec_dpo_a.checkpoint_id: 0.5, rec_sft_b.checkpoint_id: 0.5},
    )

    dag = store.get_dag(proj.project_id)
    depths = dag.generation_depths()
    assert depths[rec_base.checkpoint_id] == 0
    assert depths[rec_sft_a.checkpoint_id] == 1
    assert depths[rec_sft_b.checkpoint_id] == 1
    assert depths[rec_dpo_a.checkpoint_id] == 2
    assert depths[rec_merged.checkpoint_id] == 3

    branches = dag.enumerate_branches()
    assert len(branches) == 2
    assert all(b.contains_merge for b in branches)
    assert (
        dag.lowest_common_ancestor(rec_dpo_a.checkpoint_id, rec_sft_b.checkpoint_id)
        == rec_base.checkpoint_id
    )

    # 3. Cycle detection: self-loop, 2-cycle, and deep ancestor cycle
    with pytest.raises(CyclicDependencyError):
        store.add_relationship(
            proj.project_id,
            parent_checkpoint_id=rec_sft_a.checkpoint_id,
            child_checkpoint_id=rec_sft_a.checkpoint_id,
        )

    with pytest.raises(CyclicDependencyError):
        store.add_relationship(
            proj.project_id,
            parent_checkpoint_id=rec_merged.checkpoint_id,
            child_checkpoint_id=rec_base.checkpoint_id,
        )

    with pytest.raises(CyclicDependencyError):
        store.update_checkpoint(
            proj.project_id,
            rec_sft_a.checkpoint_id,
            parent_checkpoint_ids=[rec_merged.checkpoint_id],
        )

    # Duplicate edge rejected
    with pytest.raises(EvolutionGraphError):
        store.add_relationship(
            proj.project_id,
            parent_checkpoint_id=rec_base.checkpoint_id,
            child_checkpoint_id=rec_sft_a.checkpoint_id,
        )

    # Missing parent rejected
    with pytest.raises(EvolutionEntityNotFoundError):
        store.add_relationship(
            proj.project_id,
            parent_checkpoint_id="ckpt-nonexistent",
            child_checkpoint_id=rec_sft_a.checkpoint_id,
        )

    # 4. Duplicate checkpoint content SHA-256 detection & LoRA adapter detection
    dup = inspect_and_build_checkpoint_record(
        project_id=proj.project_id,
        source=ckpts["a"],
        version_label="Base-Duplicate",
        existing_project_checkpoints=store.list_checkpoints(proj.project_id),
        policy=policy,
        local_files_only=True,
    )
    assert dup.validation_status == CheckpointValidationStatus.DUPLICATE_CONTENT
    assert dup.duplicate_of_checkpoint_id == rec_base.checkpoint_id

    lora_rec = inspect_and_build_checkpoint_record(
        project_id=proj.project_id,
        source=ckpts["lora"],
        version_label="LoRA-r4",
        parent_checkpoint_ids=[rec_base.checkpoint_id],
        existing_project_checkpoints=store.list_checkpoints(proj.project_id),
        policy=policy,
        local_files_only=True,
    )
    assert lora_rec.transformation_type == TransformationType.LORA
    assert lora_rec.validation_status == CheckpointValidationStatus.ADAPTER_ONLY
    assert lora_rec.lora_metadata["r"] == 4
    assert lora_rec.lora_metadata["lora_alpha"] == 8

    # Quantization metadata extracted without inventing missing training metadata
    quant_rec = inspect_and_build_checkpoint_record(
        project_id=proj.project_id,
        source=ckpts["d"],
        version_label="Quant-Int4",
        parent_checkpoint_ids=[rec_dpo_a.checkpoint_id],
        existing_project_checkpoints=store.list_checkpoints(proj.project_id),
        policy=policy,
        local_files_only=True,
    )
    assert quant_rec.quantization_metadata["bits"] == 4
    assert quant_rec.training_metadata == {}

    # 5. Rename, remove checkpoint, and cascade delete project
    renamed = store.update_checkpoint(
        proj.project_id,
        rec_sft_b.checkpoint_id,
        version_label="SFT-Branch-B-Renamed",
        experiment_notes="Updated notes",
    )
    assert renamed.version_label == "SFT-Branch-B-Renamed"
    assert renamed.experiment_notes == "Updated notes"

    assert store.remove_checkpoint(proj.project_id, rec_sft_b.checkpoint_id) is True
    dag_after = EvolutionDAG(
        store.list_checkpoints(proj.project_id),
        store.list_relationships(proj.project_id),
    )
    assert rec_sft_b.checkpoint_id not in dag_after.checkpoints_by_id
    assert store.delete_project(proj.project_id) is True
