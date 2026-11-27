"""SQLite persistence & versioned migrations for Neural Evolution Tracking (Section 11)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from api.store import DEFAULT_DB_PATH
from core.evolution.dag import EvolutionDAG
from core.evolution.types import (
    CheckpointValidationStatus,
    EvolutionProject,
    EvolutionRelationship,
    RegisteredCheckpoint,
    normalize_transformation_type,
)
from core.exceptions import EvolutionEntityNotFoundError

EVOLUTION_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    (
        "0001_evolution_core_tables",
        "Create projects, checkpoints, relationships, runs, and granular evolution result tables",
        """
        CREATE TABLE IF NOT EXISTS evolution_projects (
            project_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            base_checkpoint_id TEXT,
            tags_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS evolution_checkpoints (
            checkpoint_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            version_label TEXT NOT NULL,
            model_name TEXT NOT NULL,
            source_path TEXT NOT NULL,
            resolved_path TEXT NOT NULL,
            architecture TEXT,
            model_family TEXT,
            parameter_count INTEGER NOT NULL,
            layer_count INTEGER,
            hidden_size INTEGER,
            intermediate_size INTEGER,
            num_attention_heads INTEGER,
            num_key_value_heads INTEGER,
            vocab_size INTEGER,
            tensor_count INTEGER NOT NULL,
            checkpoint_format TEXT NOT NULL,
            dtypes_json TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            sha256_fingerprint TEXT NOT NULL,
            tensor_metadata_hash TEXT NOT NULL,
            transformation_type TEXT NOT NULL,
            transformation_metadata_json TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            framework_info_json TEXT NOT NULL,
            tokenizer_metadata_json TEXT NOT NULL,
            quantization_metadata_json TEXT NOT NULL,
            training_metadata_json TEXT NOT NULL,
            lora_metadata_json TEXT NOT NULL,
            experiment_notes TEXT NOT NULL,
            validation_status TEXT NOT NULL,
            duplicate_of_checkpoint_id TEXT,
            security_notes_json TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES evolution_projects(project_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_evo_ckpts_project ON evolution_checkpoints(project_id);
        CREATE INDEX IF NOT EXISTS idx_evo_ckpts_sha ON evolution_checkpoints(project_id, sha256_fingerprint);

        CREATE TABLE IF NOT EXISTS evolution_relationships (
            relationship_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            parent_checkpoint_id TEXT NOT NULL,
            child_checkpoint_id TEXT NOT NULL,
            transformation_type TEXT NOT NULL,
            merge_weight REAL,
            transformation_metadata_json TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            UNIQUE(project_id, parent_checkpoint_id, child_checkpoint_id),
            FOREIGN KEY (project_id) REFERENCES evolution_projects(project_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_evo_rels_project ON evolution_relationships(project_id);

        CREATE TABLE IF NOT EXISTS evolution_runs (
            run_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            status TEXT NOT NULL,
            stage_message TEXT NOT NULL,
            completed_steps INTEGER NOT NULL,
            total_steps INTEGER NOT NULL,
            started_at_utc TEXT NOT NULL,
            completed_at_utc TEXT,
            config_json TEXT NOT NULL,
            provenance_json TEXT NOT NULL,
            overview_json TEXT,
            insights_json TEXT,
            warnings_json TEXT NOT NULL,
            error_json TEXT,
            FOREIGN KEY (project_id) REFERENCES evolution_projects(project_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_evo_runs_project ON evolution_runs(project_id, started_at_utc DESC);
        CREATE INDEX IF NOT EXISTS idx_evo_runs_cache ON evolution_runs(project_id, cache_key, status);

        CREATE TABLE IF NOT EXISTS evolution_tensor_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            source_checkpoint_id TEXT NOT NULL,
            target_checkpoint_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES evolution_runs(run_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_evo_tensor_run ON evolution_tensor_results(run_id);

        CREATE TABLE IF NOT EXISTS evolution_layer_results (
            run_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES evolution_runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS evolution_behavioral_cache (
            cache_id TEXT PRIMARY KEY,
            checkpoint_sha256 TEXT NOT NULL,
            suite_fingerprint TEXT NOT NULL,
            inference_hash TEXT NOT NULL,
            evaluated_at_utc TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS evolution_behavioral_results (
            run_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES evolution_runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS evolution_representation_results (
            run_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES evolution_runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS evolution_regression_events (
            event_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            first_affected_checkpoint_id TEXT NOT NULL,
            previous_checkpoint_id TEXT NOT NULL,
            transformation_type TEXT NOT NULL,
            affected_category TEXT NOT NULL,
            score_drop REAL NOT NULL,
            payload_json TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES evolution_runs(run_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_evo_reg_project ON evolution_regression_events(project_id, run_id);
        """,
    ),
)


class EvolutionStore:
    """Thread-safe SQLite store managing evolution projects, DAG checkpoints, runs, and caches."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.applied_migrations: list[dict[str, str]] = []
        self._run_migrations()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _run_migrations(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    migration_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    checksum_sha256 TEXT NOT NULL,
                    applied_at_utc TEXT NOT NULL
                );
                """
            )
            existing = {
                r["migration_id"]: r["checksum_sha256"]
                for r in conn.execute("SELECT migration_id, checksum_sha256 FROM schema_migrations").fetchall()
            }
            for mig_id, desc, sql in EVOLUTION_MIGRATIONS:
                checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
                if mig_id not in existing:
                    conn.executescript(sql)
                    conn.execute("PRAGMA foreign_keys = ON;")
                    now = dt.datetime.now(dt.UTC).isoformat()
                    conn.execute(
                        "INSERT INTO schema_migrations (migration_id, description, checksum_sha256, applied_at_utc) VALUES (?, ?, ?, ?)",
                        (mig_id, desc, checksum, now),
                    )
            rows = conn.execute(
                "SELECT migration_id, description, checksum_sha256, applied_at_utc FROM schema_migrations ORDER BY migration_id"
            ).fetchall()
            self.applied_migrations = [dict(r) for r in rows]

    def list_migrations(self) -> list[dict[str, str]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT migration_id, description, checksum_sha256, applied_at_utc FROM schema_migrations ORDER BY migration_id"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------
    def create_project(
        self,
        *,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        project_id: str | None = None,
    ) -> EvolutionProject:
        pid = project_id or f"evo-{uuid.uuid4().hex[:10]}"
        now = dt.datetime.now(dt.UTC).isoformat()
        proj = EvolutionProject(
            project_id=pid,
            name=name.strip(),
            description=description.strip(),
            base_checkpoint_id=None,
            created_at_utc=now,
            updated_at_utc=now,
            tags=list(tags or []),
            metadata=dict(metadata or {}),
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evolution_projects (
                    project_id, name, description, base_checkpoint_id,
                    tags_json, metadata_json, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proj.project_id,
                    proj.name,
                    proj.description,
                    proj.base_checkpoint_id,
                    json.dumps(proj.tags),
                    json.dumps(proj.metadata),
                    proj.created_at_utc,
                    proj.updated_at_utc,
                ),
            )
        return proj

    def get_project(self, project_id: str) -> EvolutionProject:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evolution_projects WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        if row is None:
            raise EvolutionEntityNotFoundError(
                f"Evolution project '{project_id}' not found.",
                details={"project_id": project_id},
            )
        return self._row_to_project(row)

    def list_projects(self, *, limit: int = 50, offset: int = 0) -> tuple[int, list[dict[str, Any]]]:
        with self._lock, self._connect() as conn:
            total_row = conn.execute("SELECT COUNT(*) AS cnt FROM evolution_projects").fetchone()
            total = int(total_row["cnt"]) if total_row else 0
            rows = conn.execute(
                """
                SELECT * FROM evolution_projects
                ORDER BY updated_at_utc DESC, created_at_utc DESC
                LIMIT ? OFFSET ?
                """,
                (max(1, min(limit, 200)), max(0, offset)),
            ).fetchall()

        projects = [self._row_to_project(r) for r in rows]
        enriched: list[dict[str, Any]] = []
        for p in projects:
            ckpts = self.list_checkpoints(p.project_id)
            rels = self.list_relationships(p.project_id)
            dag = EvolutionDAG(ckpts, rels) if ckpts else None
            reg_events = self.list_latest_regressions(p.project_id)
            latest_runs = self.list_runs(p.project_id, limit=1)[1]
            enriched.append(
                {
                    **p.to_dict(),
                    "checkpoint_count": len(ckpts),
                    "relationship_count": len(rels),
                    "branch_count": len(dag.enumerate_branches()) if dag else 0,
                    "regression_count": len(reg_events),
                    "latest_run": latest_runs[0] if latest_runs else None,
                }
            )
        return total, enriched

    def update_project(
        self,
        project_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        base_checkpoint_id: str | None = None,
        tags: list[str] | None = None,
    ) -> EvolutionProject:
        existing = self.get_project(project_id)
        if base_checkpoint_id is not None and base_checkpoint_id != "":
            _ = self.get_checkpoint(project_id, base_checkpoint_id)
        now = dt.datetime.now(dt.UTC).isoformat()
        new_name = name.strip() if name is not None else existing.name
        new_desc = description.strip() if description is not None else existing.description
        new_base = (
            (base_checkpoint_id or None)
            if base_checkpoint_id is not None
            else existing.base_checkpoint_id
        )
        new_tags = list(tags) if tags is not None else existing.tags

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE evolution_projects
                SET name = ?, description = ?, base_checkpoint_id = ?, tags_json = ?, updated_at_utc = ?
                WHERE project_id = ?
                """,
                (new_name, new_desc, new_base, json.dumps(new_tags), now, project_id),
            )
        return self.get_project(project_id)

    def delete_project(self, project_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM evolution_projects WHERE project_id = ?", (project_id,))
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Checkpoints & DAG Relationships
    # ------------------------------------------------------------------
    def add_checkpoint(
        self,
        ckpt: RegisteredCheckpoint,
        *,
        merge_weights: dict[str, float] | None = None,
    ) -> RegisteredCheckpoint:
        proj = self.get_project(ckpt.project_id)
        existing_ckpts = self.list_checkpoints(ckpt.project_id)
        existing_rels = self.list_relationships(ckpt.project_id)

        # Build candidate relationships for parent_checkpoint_ids and validate DAG acyclicity BEFORE writing
        now = dt.datetime.now(dt.UTC).isoformat()
        new_rels: list[EvolutionRelationship] = []
        for pid in ckpt.parent_checkpoint_ids:
            new_rels.append(
                EvolutionRelationship(
                    relationship_id=f"rel-{uuid.uuid4().hex[:10]}",
                    project_id=ckpt.project_id,
                    parent_checkpoint_id=pid,
                    child_checkpoint_id=ckpt.checkpoint_id,
                    transformation_type=ckpt.transformation_type,
                    merge_weight=(merge_weights or {}).get(pid),
                    transformation_metadata=dict(ckpt.transformation_metadata),
                    created_at_utc=now,
                )
            )

        # Validate full DAG with candidate checkpoint & edges
        _ = EvolutionDAG([*existing_ckpts, ckpt], [*existing_rels, *new_rels])

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evolution_checkpoints (
                    checkpoint_id, project_id, version_label, model_name,
                    source_path, resolved_path, architecture, model_family,
                    parameter_count, layer_count, hidden_size, intermediate_size,
                    num_attention_heads, num_key_value_heads, vocab_size, tensor_count,
                    checkpoint_format, dtypes_json, file_size_bytes,
                    sha256_fingerprint, tensor_metadata_hash, transformation_type,
                    transformation_metadata_json, created_at_utc, framework_info_json,
                    tokenizer_metadata_json, quantization_metadata_json,
                    training_metadata_json, lora_metadata_json, experiment_notes,
                    validation_status, duplicate_of_checkpoint_id, security_notes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ckpt.checkpoint_id,
                    ckpt.project_id,
                    ckpt.version_label,
                    ckpt.model_name,
                    ckpt.source_path,
                    ckpt.resolved_path,
                    ckpt.architecture,
                    ckpt.model_family,
                    ckpt.parameter_count,
                    ckpt.layer_count,
                    ckpt.hidden_size,
                    ckpt.intermediate_size,
                    ckpt.num_attention_heads,
                    ckpt.num_key_value_heads,
                    ckpt.vocab_size,
                    ckpt.tensor_count,
                    ckpt.checkpoint_format,
                    json.dumps(ckpt.dtypes_present),
                    ckpt.file_size_bytes,
                    ckpt.sha256_fingerprint,
                    ckpt.tensor_metadata_hash,
                    ckpt.transformation_type.value,
                    json.dumps(ckpt.transformation_metadata),
                    ckpt.created_at_utc,
                    json.dumps(ckpt.framework_info),
                    json.dumps(ckpt.tokenizer_metadata),
                    json.dumps(ckpt.quantization_metadata),
                    json.dumps(ckpt.training_metadata),
                    json.dumps(ckpt.lora_metadata),
                    ckpt.experiment_notes,
                    ckpt.validation_status.value,
                    ckpt.duplicate_of_checkpoint_id,
                    json.dumps(ckpt.security_notes),
                ),
            )
            for rel in new_rels:
                conn.execute(
                    """
                    INSERT INTO evolution_relationships (
                        relationship_id, project_id, parent_checkpoint_id, child_checkpoint_id,
                        transformation_type, merge_weight, transformation_metadata_json, created_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rel.relationship_id,
                        rel.project_id,
                        rel.parent_checkpoint_id,
                        rel.child_checkpoint_id,
                        rel.transformation_type.value,
                        rel.merge_weight,
                        json.dumps(rel.transformation_metadata),
                        rel.created_at_utc,
                    ),
                )

            # Auto-set project base_checkpoint_id if first root checkpoint
            new_base = proj.base_checkpoint_id
            if new_base is None and not ckpt.parent_checkpoint_ids:
                new_base = ckpt.checkpoint_id
            conn.execute(
                "UPDATE evolution_projects SET base_checkpoint_id = ?, updated_at_utc = ? WHERE project_id = ?",
                (new_base, now, ckpt.project_id),
            )

        return self.get_checkpoint(ckpt.project_id, ckpt.checkpoint_id)

    def get_checkpoint(self, project_id: str, checkpoint_id: str) -> RegisteredCheckpoint:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evolution_checkpoints WHERE project_id = ? AND checkpoint_id = ?",
                (project_id, checkpoint_id),
            ).fetchone()
            if row is None:
                raise EvolutionEntityNotFoundError(
                    f"Checkpoint '{checkpoint_id}' not found in project '{project_id}'.",
                    details={"project_id": project_id, "checkpoint_id": checkpoint_id},
                )
            p_rows = conn.execute(
                """
                SELECT parent_checkpoint_id FROM evolution_relationships
                WHERE project_id = ? AND child_checkpoint_id = ?
                ORDER BY created_at_utc ASC, relationship_id ASC
                """,
                (project_id, checkpoint_id),
            ).fetchall()
        parents = [r["parent_checkpoint_id"] for r in p_rows]
        return self._row_to_checkpoint(row, parents)

    def list_checkpoints(
        self,
        project_id: str,
        *,
        transformation_type: str | None = None,
    ) -> list[RegisteredCheckpoint]:
        _ = self.get_project(project_id)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evolution_checkpoints WHERE project_id = ? ORDER BY created_at_utc ASC, checkpoint_id ASC",
                (project_id,),
            ).fetchall()
            rel_rows = conn.execute(
                "SELECT parent_checkpoint_id, child_checkpoint_id FROM evolution_relationships WHERE project_id = ? ORDER BY created_at_utc ASC, relationship_id ASC",
                (project_id,),
            ).fetchall()
        parents_by_child: dict[str, list[str]] = {}
        for r in rel_rows:
            parents_by_child.setdefault(r["child_checkpoint_id"], []).append(r["parent_checkpoint_id"])

        items = [
            self._row_to_checkpoint(r, parents_by_child.get(r["checkpoint_id"], []))
            for r in rows
        ]
        if transformation_type:
            norm = normalize_transformation_type(transformation_type)
            items = [c for c in items if c.transformation_type == norm]
        return items

    def update_checkpoint(
        self,
        project_id: str,
        checkpoint_id: str,
        *,
        version_label: str | None = None,
        model_name: str | None = None,
        transformation_type: str | None = None,
        transformation_metadata: dict[str, Any] | None = None,
        experiment_notes: str | None = None,
        parent_checkpoint_ids: list[str] | None = None,
        validation_status: CheckpointValidationStatus | None = None,
    ) -> RegisteredCheckpoint:
        existing = self.get_checkpoint(project_id, checkpoint_id)
        new_label = version_label.strip() if version_label is not None else existing.version_label
        new_name = model_name.strip() if model_name is not None else existing.model_name
        new_trans = (
            normalize_transformation_type(transformation_type)
            if transformation_type is not None
            else existing.transformation_type
        )
        new_meta = (
            dict(transformation_metadata)
            if transformation_metadata is not None
            else existing.transformation_metadata
        )
        new_notes = experiment_notes if experiment_notes is not None else existing.experiment_notes
        new_val = validation_status if validation_status is not None else existing.validation_status

        now = dt.datetime.now(dt.UTC).isoformat()

        if parent_checkpoint_ids is not None:
            # Validate candidate parent replacement against full project DAG before mutating DB
            all_ckpts = self.list_checkpoints(project_id)
            kept_rels = [
                r
                for r in self.list_relationships(project_id)
                if r.child_checkpoint_id != checkpoint_id
            ]
            replacement_rels = [
                EvolutionRelationship(
                    relationship_id=f"rel-{uuid.uuid4().hex[:10]}",
                    project_id=project_id,
                    parent_checkpoint_id=pid,
                    child_checkpoint_id=checkpoint_id,
                    transformation_type=new_trans,
                    transformation_metadata=new_meta,
                    created_at_utc=now,
                )
                for pid in parent_checkpoint_ids
            ]
            _ = EvolutionDAG(all_ckpts, [*kept_rels, *replacement_rels])

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE evolution_checkpoints
                SET version_label = ?, model_name = ?, transformation_type = ?,
                    transformation_metadata_json = ?, experiment_notes = ?, validation_status = ?
                WHERE project_id = ? AND checkpoint_id = ?
                """,
                (
                    new_label,
                    new_name,
                    new_trans.value,
                    json.dumps(new_meta),
                    new_notes,
                    new_val.value,
                    project_id,
                    checkpoint_id,
                ),
            )
            if parent_checkpoint_ids is not None:
                conn.execute(
                    "DELETE FROM evolution_relationships WHERE project_id = ? AND child_checkpoint_id = ?",
                    (project_id, checkpoint_id),
                )
                for rel in replacement_rels:
                    conn.execute(
                        """
                        INSERT INTO evolution_relationships (
                            relationship_id, project_id, parent_checkpoint_id, child_checkpoint_id,
                            transformation_type, merge_weight, transformation_metadata_json, created_at_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rel.relationship_id,
                            rel.project_id,
                            rel.parent_checkpoint_id,
                            rel.child_checkpoint_id,
                            rel.transformation_type.value,
                            rel.merge_weight,
                            json.dumps(rel.transformation_metadata),
                            rel.created_at_utc,
                        ),
                    )
            conn.execute(
                "UPDATE evolution_projects SET updated_at_utc = ? WHERE project_id = ?",
                (now, project_id),
            )
        return self.get_checkpoint(project_id, checkpoint_id)

    def remove_checkpoint(self, project_id: str, checkpoint_id: str) -> bool:
        proj = self.get_project(project_id)
        _ = self.get_checkpoint(project_id, checkpoint_id)
        now = dt.datetime.now(dt.UTC).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM evolution_relationships WHERE project_id = ? AND (parent_checkpoint_id = ? OR child_checkpoint_id = ?)",
                (project_id, checkpoint_id, checkpoint_id),
            )
            cur = conn.execute(
                "DELETE FROM evolution_checkpoints WHERE project_id = ? AND checkpoint_id = ?",
                (project_id, checkpoint_id),
            )
            if proj.base_checkpoint_id == checkpoint_id:
                conn.execute(
                    "UPDATE evolution_projects SET base_checkpoint_id = NULL, updated_at_utc = ? WHERE project_id = ?",
                    (now, project_id),
                )
            else:
                conn.execute(
                    "UPDATE evolution_projects SET updated_at_utc = ? WHERE project_id = ?",
                    (now, project_id),
                )
            return cur.rowcount > 0

    def list_relationships(self, project_id: str) -> list[EvolutionRelationship]:
        _ = self.get_project(project_id)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evolution_relationships WHERE project_id = ? ORDER BY created_at_utc ASC, relationship_id ASC",
                (project_id,),
            ).fetchall()
        return [
            EvolutionRelationship(
                relationship_id=r["relationship_id"],
                project_id=r["project_id"],
                parent_checkpoint_id=r["parent_checkpoint_id"],
                child_checkpoint_id=r["child_checkpoint_id"],
                transformation_type=normalize_transformation_type(r["transformation_type"]),
                merge_weight=r["merge_weight"],
                transformation_metadata=json.loads(r["transformation_metadata_json"]),
                created_at_utc=r["created_at_utc"],
            )
            for r in rows
        ]

    def add_relationship(
        self,
        project_id: str,
        *,
        parent_checkpoint_id: str,
        child_checkpoint_id: str,
        transformation_type: str | None = None,
        merge_weight: float | None = None,
        transformation_metadata: dict[str, Any] | None = None,
    ) -> EvolutionRelationship:
        ckpts = self.list_checkpoints(project_id)
        rels = self.list_relationships(project_id)
        dag = EvolutionDAG(ckpts, rels)
        dag.check_can_add_edge(parent_checkpoint_id, child_checkpoint_id)

        child_ckpt = dag.checkpoints_by_id[child_checkpoint_id]
        eff_trans = (
            normalize_transformation_type(transformation_type)
            if transformation_type is not None
            else child_ckpt.transformation_type
        )
        now = dt.datetime.now(dt.UTC).isoformat()
        rel = EvolutionRelationship(
            relationship_id=f"rel-{uuid.uuid4().hex[:10]}",
            project_id=project_id,
            parent_checkpoint_id=parent_checkpoint_id,
            child_checkpoint_id=child_checkpoint_id,
            transformation_type=eff_trans,
            merge_weight=merge_weight,
            transformation_metadata=dict(transformation_metadata or {}),
            created_at_utc=now,
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evolution_relationships (
                    relationship_id, project_id, parent_checkpoint_id, child_checkpoint_id,
                    transformation_type, merge_weight, transformation_metadata_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rel.relationship_id,
                    rel.project_id,
                    rel.parent_checkpoint_id,
                    rel.child_checkpoint_id,
                    rel.transformation_type.value,
                    rel.merge_weight,
                    json.dumps(rel.transformation_metadata),
                    rel.created_at_utc,
                ),
            )
            conn.execute(
                "UPDATE evolution_projects SET updated_at_utc = ? WHERE project_id = ?",
                (now, project_id),
            )
        return rel

    def remove_relationship(self, project_id: str, relationship_id: str) -> bool:
        now = dt.datetime.now(dt.UTC).isoformat()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM evolution_relationships WHERE project_id = ? AND relationship_id = ?",
                (project_id, relationship_id),
            )
            conn.execute(
                "UPDATE evolution_projects SET updated_at_utc = ? WHERE project_id = ?",
                (now, project_id),
            )
            return cur.rowcount > 0

    def get_dag(self, project_id: str) -> EvolutionDAG:
        ckpts = self.list_checkpoints(project_id)
        rels = self.list_relationships(project_id)
        return EvolutionDAG(ckpts, rels)

    # ------------------------------------------------------------------
    # Behavioral Per-Checkpoint Content Cache
    # ------------------------------------------------------------------
    def get_cached_checkpoint_eval(
        self,
        checkpoint_sha256: str,
        suite_fingerprint: str,
        inference_hash: str,
    ) -> dict[str, Any] | None:
        cache_id = hashlib.sha256(
            f"{checkpoint_sha256}:{suite_fingerprint}:{inference_hash}".encode()
        ).hexdigest()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM evolution_behavioral_cache WHERE cache_id = ?",
                (cache_id,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def save_cached_checkpoint_eval(
        self,
        checkpoint_sha256: str,
        suite_fingerprint: str,
        inference_hash: str,
        payload: dict[str, Any],
    ) -> None:
        cache_id = hashlib.sha256(
            f"{checkpoint_sha256}:{suite_fingerprint}:{inference_hash}".encode()
        ).hexdigest()
        now = dt.datetime.now(dt.UTC).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evolution_behavioral_cache (
                    cache_id, checkpoint_sha256, suite_fingerprint, inference_hash, evaluated_at_utc, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_id,
                    checkpoint_sha256,
                    suite_fingerprint,
                    inference_hash,
                    now,
                    json.dumps(payload, allow_nan=False),
                ),
            )

    # ------------------------------------------------------------------
    # Evolution Runs & Reproducible Result Storage
    # ------------------------------------------------------------------
    def find_cached_run(self, project_id: str, cache_key: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id FROM evolution_runs
                WHERE project_id = ? AND cache_key = ? AND status = 'completed'
                ORDER BY completed_at_utc DESC
                LIMIT 1
                """,
                (project_id, cache_key),
            ).fetchone()
        if row is None:
            return None
        return self.get_run(str(row["run_id"]))

    def create_run(
        self,
        *,
        run_id: str,
        project_id: str,
        cache_key: str,
        config: dict[str, Any],
        provenance: dict[str, Any],
        total_steps: int,
    ) -> None:
        now = dt.datetime.now(dt.UTC).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evolution_runs (
                    run_id, project_id, cache_key, status, stage_message,
                    completed_steps, total_steps, started_at_utc, completed_at_utc,
                    config_json, provenance_json, overview_json, insights_json, warnings_json, error_json
                ) VALUES (?, ?, ?, 'running', 'Initializing evolution analysis', 0, ?, ?, NULL, ?, ?, NULL, NULL, '[]', NULL)
                """,
                (
                    run_id,
                    project_id,
                    cache_key,
                    total_steps,
                    now,
                    json.dumps(config),
                    json.dumps(provenance),
                ),
            )

    def update_run_progress(
        self,
        run_id: str,
        stage_message: str,
        completed_steps: int,
        total_steps: int,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE evolution_runs
                SET stage_message = ?, completed_steps = ?, total_steps = ?
                WHERE run_id = ?
                """,
                (stage_message, completed_steps, total_steps, run_id),
            )

    def save_completed_run(
        self,
        *,
        run_id: str,
        project_id: str,
        provenance: dict[str, Any],
        overview: dict[str, Any],
        tensor_transitions: list[dict[str, Any]],
        layer_evolution: dict[str, Any],
        behavioral_evolution: dict[str, Any] | None,
        representation_evolution: dict[str, Any] | None,
        regression_events: list[dict[str, Any]],
        insights: list[dict[str, Any]],
        warnings: list[str],
    ) -> None:
        now = dt.datetime.now(dt.UTC).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE evolution_runs
                SET status = 'completed',
                    stage_message = 'Completed',
                    completed_steps = total_steps,
                    completed_at_utc = ?,
                    provenance_json = ?,
                    overview_json = ?,
                    insights_json = ?,
                    warnings_json = ?
                WHERE run_id = ?
                """,
                (
                    now,
                    json.dumps(provenance, allow_nan=False),
                    json.dumps(overview, allow_nan=False),
                    json.dumps(insights, allow_nan=False),
                    json.dumps(warnings),
                    run_id,
                ),
            )
            conn.execute("DELETE FROM evolution_tensor_results WHERE run_id = ?", (run_id,))
            for tr in tensor_transitions:
                conn.execute(
                    """
                    INSERT INTO evolution_tensor_results (
                        run_id, project_id, scope, source_checkpoint_id, target_checkpoint_id, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        project_id,
                        tr["scope"],
                        tr["source_checkpoint_id"],
                        tr["target_checkpoint_id"],
                        json.dumps(tr, allow_nan=False),
                    ),
                )

            conn.execute(
                "INSERT OR REPLACE INTO evolution_layer_results (run_id, project_id, payload_json) VALUES (?, ?, ?)",
                (run_id, project_id, json.dumps(layer_evolution, allow_nan=False)),
            )

            if behavioral_evolution is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO evolution_behavioral_results (run_id, project_id, payload_json) VALUES (?, ?, ?)",
                    (run_id, project_id, json.dumps(behavioral_evolution, allow_nan=False)),
                )

            if representation_evolution is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO evolution_representation_results (run_id, project_id, payload_json) VALUES (?, ?, ?)",
                    (run_id, project_id, json.dumps(representation_evolution, allow_nan=False)),
                )

            conn.execute("DELETE FROM evolution_regression_events WHERE run_id = ?", (run_id,))
            for ev in regression_events:
                conn.execute(
                    """
                    INSERT INTO evolution_regression_events (
                        event_id, run_id, project_id, first_affected_checkpoint_id,
                        previous_checkpoint_id, transformation_type, affected_category,
                        score_drop, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ev["event_id"],
                        run_id,
                        project_id,
                        ev["first_affected_checkpoint_id"],
                        ev["previous_checkpoint_id"],
                        ev["transformation_type"],
                        ev["affected_category"],
                        float(ev["score_drop"]),
                        json.dumps(ev, allow_nan=False),
                    ),
                )

            conn.execute(
                "UPDATE evolution_projects SET updated_at_utc = ? WHERE project_id = ?",
                (now, project_id),
            )

    def save_failed_run(
        self,
        run_id: str,
        error_dict: dict[str, Any],
        *,
        cancelled: bool = False,
    ) -> None:
        status = "cancelled" if cancelled else "failed"
        msg = str(error_dict.get("message", status))[:240]
        now = dt.datetime.now(dt.UTC).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE evolution_runs
                SET status = ?, stage_message = ?, completed_at_utc = ?, error_json = ?
                WHERE run_id = ?
                """,
                (status, msg, now, json.dumps(error_dict), run_id),
            )

    def list_runs(
        self,
        project_id: str,
        *,
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[int, list[dict[str, Any]]]:
        _ = self.get_project(project_id)
        with self._lock, self._connect() as conn:
            cnt_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM evolution_runs WHERE project_id = ?",
                (project_id,),
            ).fetchone()
            total = int(cnt_row["cnt"]) if cnt_row else 0
            rows = conn.execute(
                """
                SELECT run_id, project_id, cache_key, status, stage_message,
                       completed_steps, total_steps, started_at_utc, completed_at_utc,
                       config_json, overview_json, warnings_json, error_json
                FROM evolution_runs
                WHERE project_id = ?
                ORDER BY started_at_utc DESC
                LIMIT ? OFFSET ?
                """,
                (project_id, max(1, min(limit, 200)), max(0, offset)),
            ).fetchall()

        summaries = [
            {
                "run_id": r["run_id"],
                "project_id": r["project_id"],
                "cache_key": r["cache_key"],
                "status": r["status"],
                "stage_message": r["stage_message"],
                "completed_steps": r["completed_steps"],
                "total_steps": r["total_steps"],
                "started_at_utc": r["started_at_utc"],
                "completed_at_utc": r["completed_at_utc"],
                "config": json.loads(r["config_json"]),
                "overview": json.loads(r["overview_json"]) if r["overview_json"] else None,
                "warnings": json.loads(r["warnings_json"]),
                "error": json.loads(r["error_json"]) if r["error_json"] else None,
            }
            for r in rows
        ]
        return total, summaries

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM evolution_runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise EvolutionEntityNotFoundError(
                    f"Evolution analysis run '{run_id}' not found.",
                    details={"run_id": run_id},
                )
            t_rows = conn.execute(
                "SELECT payload_json FROM evolution_tensor_results WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            ).fetchall()
            l_row = conn.execute(
                "SELECT payload_json FROM evolution_layer_results WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            b_row = conn.execute(
                "SELECT payload_json FROM evolution_behavioral_results WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            rep_row = conn.execute(
                "SELECT payload_json FROM evolution_representation_results WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            reg_rows = conn.execute(
                "SELECT payload_json FROM evolution_regression_events WHERE run_id = ? ORDER BY score_drop DESC",
                (run_id,),
            ).fetchall()

        project_id = str(row["project_id"])
        proj = self.get_project(project_id)
        ckpts = self.list_checkpoints(project_id)
        dag = self.get_dag(project_id)

        return {
            "run_id": row["run_id"],
            "project_id": project_id,
            "cache_key": row["cache_key"],
            "status": row["status"],
            "stage_message": row["stage_message"],
            "completed_steps": row["completed_steps"],
            "total_steps": row["total_steps"],
            "started_at_utc": row["started_at_utc"],
            "completed_at_utc": row["completed_at_utc"],
            "config": json.loads(row["config_json"]),
            "provenance": json.loads(row["provenance_json"]),
            "project": proj.to_dict(),
            "checkpoints": [c.to_dict() for c in ckpts],
            "graph": dag.to_graph_payload(proj.base_checkpoint_id),
            "overview": json.loads(row["overview_json"]) if row["overview_json"] else None,
            "tensor_transitions": [json.loads(r["payload_json"]) for r in t_rows],
            "layer_evolution": json.loads(l_row["payload_json"]) if l_row else None,
            "behavioral_evolution": json.loads(b_row["payload_json"]) if b_row else None,
            "representation_evolution": json.loads(rep_row["payload_json"]) if rep_row else None,
            "regression_events": [json.loads(r["payload_json"]) for r in reg_rows],
            "insights": json.loads(row["insights_json"]) if row["insights_json"] else [],
            "warnings": json.loads(row["warnings_json"]),
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
        }

    def list_latest_regressions(
        self,
        project_id: str,
        *,
        category: str | None = None,
        transformation_type: str | None = None,
    ) -> list[dict[str, Any]]:
        _ = self.get_project(project_id)
        with self._lock, self._connect() as conn:
            latest = conn.execute(
                """
                SELECT run_id FROM evolution_runs
                WHERE project_id = ? AND status = 'completed'
                ORDER BY completed_at_utc DESC
                LIMIT 1
                """,
                (project_id,),
            ).fetchone()
            if latest is None:
                return []
            rows = conn.execute(
                """
                SELECT payload_json FROM evolution_regression_events
                WHERE run_id = ?
                ORDER BY score_drop DESC
                """,
                (latest["run_id"],),
            ).fetchall()

        events = [json.loads(r["payload_json"]) for r in rows]
        if category:
            events = [e for e in events if e.get("affected_category") == category]
        if transformation_type:
            norm = normalize_transformation_type(transformation_type).value
            events = [e for e in events if e.get("transformation_type") == norm]
        return events

    def global_analytics_summary(self) -> dict[str, Any]:
        """Compute cross-project analytics counts for the Evolution Dashboard overview."""
        with self._lock, self._connect() as conn:
            proj_cnt = int(conn.execute("SELECT COUNT(*) FROM evolution_projects").fetchone()[0])
            ckpt_cnt = int(conn.execute("SELECT COUNT(*) FROM evolution_checkpoints").fetchone()[0])
            rel_cnt = int(conn.execute("SELECT COUNT(*) FROM evolution_relationships").fetchone()[0])
            run_cnt = int(conn.execute("SELECT COUNT(*) FROM evolution_runs WHERE status = 'completed'").fetchone()[0])

        _, proj_list = self.list_projects(limit=200)
        total_branches = sum(int(p["branch_count"]) for p in proj_list)
        total_regressions = sum(int(p["regression_count"]) for p in proj_list)
        return {
            "total_projects": proj_cnt,
            "total_checkpoints": ckpt_cnt,
            "total_transformations": rel_cnt,
            "total_branches": total_branches,
            "total_detected_regressions": total_regressions,
            "completed_analysis_runs": run_cnt,
        }

    # ------------------------------------------------------------------
    # Row Mappers
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_project(row: sqlite3.Row) -> EvolutionProject:
        return EvolutionProject(
            project_id=row["project_id"],
            name=row["name"],
            description=row["description"],
            base_checkpoint_id=row["base_checkpoint_id"],
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
            tags=json.loads(row["tags_json"]),
            metadata=json.loads(row["metadata_json"]),
        )

    @staticmethod
    def _row_to_checkpoint(row: sqlite3.Row, parent_ids: list[str]) -> RegisteredCheckpoint:
        return RegisteredCheckpoint(
            checkpoint_id=row["checkpoint_id"],
            project_id=row["project_id"],
            version_label=row["version_label"],
            model_name=row["model_name"],
            source_path=row["source_path"],
            resolved_path=row["resolved_path"],
            architecture=row["architecture"],
            model_family=row["model_family"],
            parameter_count=int(row["parameter_count"]),
            layer_count=row["layer_count"],
            hidden_size=row["hidden_size"],
            intermediate_size=row["intermediate_size"],
            num_attention_heads=row["num_attention_heads"],
            num_key_value_heads=row["num_key_value_heads"],
            vocab_size=row["vocab_size"],
            tensor_count=int(row["tensor_count"]),
            checkpoint_format=row["checkpoint_format"],
            dtypes_present=json.loads(row["dtypes_json"]),
            file_size_bytes=int(row["file_size_bytes"]),
            sha256_fingerprint=row["sha256_fingerprint"],
            tensor_metadata_hash=row["tensor_metadata_hash"],
            parent_checkpoint_ids=parent_ids,
            transformation_type=normalize_transformation_type(row["transformation_type"]),
            transformation_metadata=json.loads(row["transformation_metadata_json"]),
            created_at_utc=row["created_at_utc"],
            framework_info=json.loads(row["framework_info_json"]),
            tokenizer_metadata=json.loads(row["tokenizer_metadata_json"]),
            quantization_metadata=json.loads(row["quantization_metadata_json"]),
            training_metadata=json.loads(row["training_metadata_json"]),
            lora_metadata=json.loads(row["lora_metadata_json"]),
            experiment_notes=row["experiment_notes"],
            validation_status=CheckpointValidationStatus(row["validation_status"]),
            duplicate_of_checkpoint_id=row["duplicate_of_checkpoint_id"],
            security_notes=json.loads(row["security_notes_json"]),
        )
