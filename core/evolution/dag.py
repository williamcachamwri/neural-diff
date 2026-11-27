"""Directed Acyclic Graph (DAG) engine for branching & merged model evolution histories."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from core.evolution.types import EvolutionRelationship, RegisteredCheckpoint
from core.exceptions import CyclicDependencyError, EvolutionEntityNotFoundError, EvolutionGraphError


@dataclass(frozen=True)
class BranchLineage:
    """A distinct root-to-leaf evolution path through the checkpoint DAG."""

    branch_id: str
    root_checkpoint_id: str
    leaf_checkpoint_id: str
    checkpoint_ids: list[str]
    version_labels: list[str]
    contains_merge: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "root_checkpoint_id": self.root_checkpoint_id,
            "leaf_checkpoint_id": self.leaf_checkpoint_id,
            "checkpoint_ids": list(self.checkpoint_ids),
            "version_labels": list(self.version_labels),
            "contains_merge": self.contains_merge,
        }


class EvolutionDAG:
    """Validated DAG representation over registered checkpoints and parent-child relationships."""

    def __init__(
        self,
        checkpoints: list[RegisteredCheckpoint],
        relationships: list[EvolutionRelationship],
    ) -> None:
        self.checkpoints_by_id: dict[str, RegisteredCheckpoint] = {
            c.checkpoint_id: c for c in checkpoints
        }
        self.relationships: list[EvolutionRelationship] = list(relationships)
        self.parents_of: dict[str, list[str]] = {c.checkpoint_id: [] for c in checkpoints}
        self.children_of: dict[str, list[str]] = {c.checkpoint_id: [] for c in checkpoints}
        self.edge_by_pair: dict[tuple[str, str], EvolutionRelationship] = {}

        for rel in relationships:
            p_id = rel.parent_checkpoint_id
            c_id = rel.child_checkpoint_id
            if p_id not in self.checkpoints_by_id:
                raise EvolutionEntityNotFoundError(
                    f"Parent checkpoint '{p_id}' is missing from project registry.",
                    details={"parent_checkpoint_id": p_id, "child_checkpoint_id": c_id},
                )
            if c_id not in self.checkpoints_by_id:
                raise EvolutionEntityNotFoundError(
                    f"Child checkpoint '{c_id}' is missing from project registry.",
                    details={"parent_checkpoint_id": p_id, "child_checkpoint_id": c_id},
                )
            if p_id == c_id:
                raise CyclicDependencyError(
                    f"Self-referential dependency detected on checkpoint '{p_id}'.",
                    details={"cycle": [p_id, p_id]},
                )
            if p_id not in self.parents_of[c_id]:
                self.parents_of[c_id].append(p_id)
            if c_id not in self.children_of[p_id]:
                self.children_of[p_id].append(c_id)
            self.edge_by_pair[(p_id, c_id)] = rel

        self.validate_acyclic()

    def validate_acyclic(self) -> None:
        """Detect circular dependencies using 3-color DFS and report the exact cycle path."""
        # 0 = unvisited, 1 = visiting (on stack), 2 = visited
        state: dict[str, int] = dict.fromkeys(self.checkpoints_by_id, 0)
        stack: list[str] = []

        def _dfs(node_id: str) -> None:
            state[node_id] = 1
            stack.append(node_id)
            for child_id in self.children_of.get(node_id, []):
                st = state.get(child_id, 0)
                if st == 1:
                    cycle_start = stack.index(child_id)
                    cycle_path = [*stack[cycle_start:], child_id]
                    raise CyclicDependencyError(
                        f"Circular dependency detected in evolution history: {' -> '.join(cycle_path)}",
                        details={"cycle": cycle_path},
                    )
                if st == 0:
                    _dfs(child_id)
            stack.pop()
            state[node_id] = 2

        for cid in sorted(self.checkpoints_by_id):
            if state[cid] == 0:
                _dfs(cid)

    def check_can_add_edge(self, parent_id: str, child_id: str) -> None:
        """Verify adding `parent_id -> child_id` preserves DAG validity before persisting."""
        if parent_id not in self.checkpoints_by_id:
            raise EvolutionEntityNotFoundError(
                f"Parent checkpoint '{parent_id}' does not exist in this project.",
                details={"parent_checkpoint_id": parent_id},
            )
        if child_id not in self.checkpoints_by_id:
            raise EvolutionEntityNotFoundError(
                f"Child checkpoint '{child_id}' does not exist in this project.",
                details={"child_checkpoint_id": child_id},
            )
        if parent_id == child_id:
            raise CyclicDependencyError(
                f"Checkpoint '{parent_id}' cannot be its own parent.",
                details={"cycle": [parent_id, child_id]},
            )
        if (parent_id, child_id) in self.edge_by_pair:
            raise EvolutionGraphError(
                f"Relationship '{parent_id} -> {child_id}' already exists.",
                code="DUPLICATE_RELATIONSHIP_ERROR",
                http_status=409,
            )

        # Adding parent -> child creates a cycle iff `parent_id` is reachable from `child_id`
        anc_path = self.shortest_directed_path(child_id, parent_id)
        if anc_path is not None:
            cycle = [*anc_path, child_id]
            raise CyclicDependencyError(
                f"Adding '{parent_id} -> {child_id}' creates a cycle: {' -> '.join(cycle)}",
                details={"cycle": cycle},
            )

    def topological_order(self) -> list[str]:
        """Deterministic Kahn's topological sort breaking ties by creation timestamp then ID."""
        in_deg: dict[str, int] = {cid: len(ps) for cid, ps in self.parents_of.items()}

        def _sort_key(cid: str) -> tuple[str, str]:
            ckpt = self.checkpoints_by_id[cid]
            return (ckpt.created_at_utc, cid)

        ready = sorted([cid for cid, d in in_deg.items() if d == 0], key=_sort_key)
        queue: deque[str] = deque(ready)
        ordered: list[str] = []

        while queue:
            curr = queue.popleft()
            ordered.append(curr)
            newly_ready: list[str] = []
            for ch in self.children_of.get(curr, []):
                in_deg[ch] -= 1
                if in_deg[ch] == 0:
                    newly_ready.append(ch)
            for ch in sorted(newly_ready, key=_sort_key):
                queue.append(ch)

        if len(ordered) != len(self.checkpoints_by_id):
            # Defensive fallback; validate_acyclic() already raises exact cycle
            self.validate_acyclic()
        return ordered

    def roots(self) -> list[str]:
        return [cid for cid in self.topological_order() if not self.parents_of[cid]]

    def leaves(self) -> list[str]:
        return [cid for cid in self.topological_order() if not self.children_of[cid]]

    def generation_depths(self) -> dict[str, int]:
        """Compute longest-path generation index (`0` for root models, `1 + max(parent)` downstream)."""
        depths: dict[str, int] = {}
        for cid in self.topological_order():
            ps = self.parents_of[cid]
            depths[cid] = 0 if not ps else 1 + max(depths[p] for p in ps)
        return depths

    def shortest_directed_path(self, source_id: str, target_id: str) -> list[str] | None:
        """Return shortest directed path `source_id -> ... -> target_id`, or `None` if unreachable."""
        if source_id not in self.checkpoints_by_id or target_id not in self.checkpoints_by_id:
            return None
        if source_id == target_id:
            return [source_id]
        q: deque[list[str]] = deque([[source_id]])
        visited = {source_id}
        while q:
            path = q.popleft()
            last = path[-1]
            for nxt in self.children_of.get(last, []):
                if nxt == target_id:
                    return [*path, nxt]
                if nxt not in visited:
                    visited.add(nxt)
                    q.append([*path, nxt])
        return None

    def ancestors_of(self, checkpoint_id: str) -> set[str]:
        visited: set[str] = set()
        stack = list(self.parents_of.get(checkpoint_id, []))
        while stack:
            cur = stack.pop()
            if cur not in visited:
                visited.add(cur)
                stack.extend(self.parents_of.get(cur, []))
        return visited

    def descendants_of(self, checkpoint_id: str) -> set[str]:
        visited: set[str] = set()
        stack = list(self.children_of.get(checkpoint_id, []))
        while stack:
            cur = stack.pop()
            if cur not in visited:
                visited.add(cur)
                stack.extend(self.children_of.get(cur, []))
        return visited

    def lowest_common_ancestor(self, id_a: str, id_b: str) -> str | None:
        """Find deepest shared ancestor of `id_a` and `id_b` (including self if one is ancestor)."""
        anc_a = self.ancestors_of(id_a) | {id_a}
        anc_b = self.ancestors_of(id_b) | {id_b}
        shared = anc_a & anc_b
        if not shared:
            return None
        depths = self.generation_depths()
        return max(shared, key=lambda cid: (depths.get(cid, 0), cid))

    def enumerate_branches(self, max_paths: int = 128) -> list[BranchLineage]:
        """Enumerate distinct root-to-leaf evolution lineages through the DAG."""
        lineages: list[BranchLineage] = []
        roots = self.roots()

        def _walk(current: str, path: list[str]) -> None:
            if len(lineages) >= max_paths:
                return
            children = self.children_of.get(current, [])
            if not children:
                has_merge = any(len(self.parents_of[cid]) > 1 for cid in path)
                labels = [self.checkpoints_by_id[cid].version_label for cid in path]
                idx = len(lineages) + 1
                lineages.append(
                    BranchLineage(
                        branch_id=f"branch-{idx:02d}",
                        root_checkpoint_id=path[0],
                        leaf_checkpoint_id=path[-1],
                        checkpoint_ids=list(path),
                        version_labels=labels,
                        contains_merge=has_merge,
                    )
                )
                return
            for ch in children:
                _walk(ch, [*path, ch])

        for r in roots:
            _walk(r, [r])
        return lineages

    def to_graph_payload(self, base_checkpoint_id: str | None = None) -> dict[str, Any]:
        """Serialize complete topological graph, node coordinates/depths, edges, and branches."""
        topo = self.topological_order()
        depths = self.generation_depths()
        branches = self.enumerate_branches()
        effective_base = base_checkpoint_id or (self.roots()[0] if self.roots() else None)

        # Assign lane index per depth level for clean horizontal/vertical DAG layout
        by_depth: dict[int, list[str]] = {}
        for cid in topo:
            by_depth.setdefault(depths[cid], []).append(cid)

        nodes: list[dict[str, Any]] = []
        for cid in topo:
            ckpt = self.checkpoints_by_id[cid]
            d = depths[cid]
            lane_peers = by_depth[d]
            lane_index = lane_peers.index(cid)
            nodes.append(
                {
                    "checkpoint_id": cid,
                    "version_label": ckpt.version_label,
                    "model_name": ckpt.model_name,
                    "transformation_type": ckpt.transformation_type.value,
                    "generation_depth": d,
                    "lane_index": lane_index,
                    "lane_count": len(lane_peers),
                    "is_root": len(self.parents_of[cid]) == 0,
                    "is_leaf": len(self.children_of[cid]) == 0,
                    "is_merge_node": len(self.parents_of[cid]) > 1,
                    "is_branch_point": len(self.children_of[cid]) > 1,
                    "is_base_model": cid == effective_base,
                    "parent_ids": list(self.parents_of[cid]),
                    "child_ids": list(self.children_of[cid]),
                    "parameter_count": ckpt.parameter_count,
                    "layer_count": ckpt.layer_count,
                    "hidden_size": ckpt.hidden_size,
                    "dtypes_present": ckpt.dtypes_present,
                    "sha256_fingerprint": ckpt.sha256_fingerprint,
                    "validation_status": ckpt.validation_status.value,
                    "duplicate_of_checkpoint_id": ckpt.duplicate_of_checkpoint_id,
                    "created_at_utc": ckpt.created_at_utc,
                }
            )

        edges = [rel.to_dict() for rel in self.relationships]
        return {
            "base_checkpoint_id": effective_base,
            "topological_order": topo,
            "root_checkpoint_ids": self.roots(),
            "leaf_checkpoint_ids": self.leaves(),
            "branch_count": len(branches),
            "branches": [b.to_dict() for b in branches],
            "nodes": nodes,
            "edges": edges,
        }
