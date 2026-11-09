"""Configuration Settings Grid, lifecycle ledger, AST zero-magic-literal gate, and domain-agnostic naming gate."""

from __future__ import annotations

import ast
import json
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALLOWED_NUMERIC_IDENTITIES: frozenset[int | float] = frozenset({0, 1, -1, 0.0, 1.0})

DEFAULT_LIFECYCLE_STAGES: tuple[str, ...] = (
    "represented",
    "applicable",
    "proposed",
    "dispatched",
    "evaluated",
    "audited",
    "promoted",
)

FORBIDDEN_DOMAIN_PATTERN = re.compile(
    r"(real_estate|mortgage|healthcare|hospitality|fintech|ecommerce|parcel_permit|"
    r"stock_picker|medical_triage|retail_cart|insurance_claim|legal_brief)",
    re.IGNORECASE,
)


def default_grid_path() -> Path:
    """Resolve the repository Configuration Settings Grid JSON path."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config" / "latent_prism_grid.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not locate config/latent_prism_grid.json")


def round_nested_floats(payload: Any, digits: int) -> Any:
    """Recursively round float scalars for compact deterministic JSON serialization."""
    if isinstance(payload, bool) or payload is None:
        return payload
    if isinstance(payload, float):
        return round(payload, digits)
    if isinstance(payload, int):
        return payload
    if isinstance(payload, dict):
        return {str(k): round_nested_floats(v, digits) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [round_nested_floats(v, digits) for v in payload]
    return payload


@dataclass(frozen=True)
class ParameterBounds:
    """Runtime clamping and upper-boundary saturation metadata for a NodeParameter axis."""

    min_value: int | float
    max_value: int | float
    expansion_step: int | float

    def clamp(self, value: Any) -> Any:
        if isinstance(value, (bool, str)):
            return value
        if value < self.min_value:
            return self.min_value
        if value > self.max_value:
            return self.max_value
        return value

    def is_upper_saturated(self, value: Any) -> bool:
        if isinstance(value, (bool, str)) or self.expansion_step <= 0:
            return False
        return bool(value >= self.max_value)


@dataclass(frozen=True)
class NodeParameter:
    """Single tunable parameter axis inside NodeGrid."""

    axis_id: str
    kind: str
    default: Any
    bounds: ParameterBounds
    choices: tuple[Any, ...] = ()
    low: int | None = None
    high: int | None = None
    step: int = 1

    @classmethod
    def from_spec(cls, raw: dict[str, Any]) -> NodeParameter:
        b_raw = raw["bounds"]
        bounds = ParameterBounds(
            min_value=b_raw["min_value"],
            max_value=b_raw["max_value"],
            expansion_step=b_raw["expansion_step"],
        )
        return cls(
            axis_id=str(raw["axis_id"]),
            kind=str(raw["kind"]),
            default=raw["default"],
            bounds=bounds,
            choices=tuple(raw.get("choices", ())),
            low=raw.get("low"),
            high=raw.get("high"),
            step=int(raw.get("step", 1)),
        )

    def validate_value(self, value: Any) -> Any:
        clamped = self.bounds.clamp(value)
        if self.kind == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"Axis {self.axis_id} requires boolean value")
            return value
        if self.kind == "choice":
            if clamped not in self.choices:
                raise ValueError(f"Axis {self.axis_id}={clamped} outside choices {self.choices}")
            return clamped
        if self.kind == "integer_range":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Axis {self.axis_id} requires integer value")
            if self.low is not None and clamped < self.low:
                raise ValueError(f"Axis {self.axis_id}={clamped} below low={self.low}")
            if self.high is not None and clamped > self.high:
                raise ValueError(f"Axis {self.axis_id}={clamped} above high={self.high}")
            return int(clamped)
        raise ValueError(f"Unsupported parameter axis kind: {self.kind}")

    def expand_upper_boundary(self) -> NodeParameter:
        if self.bounds.expansion_step <= 0:
            return self
        next_bounds = ParameterBounds(
            min_value=self.bounds.min_value,
            max_value=self.bounds.max_value + self.bounds.expansion_step,
            expansion_step=self.bounds.expansion_step,
        )
        if self.kind == "integer_range" and self.high is not None:
            next_high = min(self.high + int(self.bounds.expansion_step), int(next_bounds.max_value))
            return NodeParameter(
                axis_id=self.axis_id,
                kind=self.kind,
                default=self.default,
                bounds=next_bounds,
                choices=self.choices,
                low=self.low,
                high=next_high,
                step=self.step,
            )
        if self.kind == "choice" and self.choices and not isinstance(self.choices[0], str):
            top_val = max(self.choices)
            candidate = min(top_val + self.bounds.expansion_step, next_bounds.max_value)
            next_choices = self.choices if candidate in self.choices else (*self.choices, candidate)
            return NodeParameter(
                axis_id=self.axis_id,
                kind=self.kind,
                default=self.default,
                bounds=next_bounds,
                choices=next_choices,
                low=self.low,
                high=self.high,
                step=self.step,
            )
        return self


@dataclass
class OperationCostLedger:
    """Empirical cost-and-quality store backing contextual regime cell resolution."""

    observations: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        regime_id: str,
        workload_signature: str,
        cell_id: str,
        quality_score: float,
        latency_us: float,
        payload_bytes: int,
    ) -> None:
        self.observations.append(
            {
                "regime_id": regime_id,
                "workload_signature": workload_signature,
                "cell_id": cell_id,
                "quality_score": float(quality_score),
                "latency_us": float(latency_us),
                "payload_bytes": int(payload_bytes),
            }
        )

    def best_for_signature(self, workload_signature: str) -> dict[str, Any] | None:
        matches = [row for row in self.observations if row["workload_signature"] == workload_signature]
        if not matches:
            return None
        return max(matches, key=lambda row: (row["quality_score"], -row["latency_us"]))


@dataclass
class GridLedger:
    """Tracks every configuration cell across lifecycle stages with disjoint maker/checker gating."""

    stages: tuple[str, ...] = DEFAULT_LIFECYCLE_STAGES
    cells: dict[str, dict[str, Any]] = field(default_factory=dict)

    def register_cell(self, cell_id: str, regime_id: str, values: dict[str, Any], maker_id: str) -> None:
        self.cells[cell_id] = {
            "cell_id": cell_id,
            "regime_id": regime_id,
            "values": dict(values),
            "maker_id": maker_id,
            "checker_id": None,
            "stage_index": 0,
            "stage": self.stages[0],
            "history": [self.stages[0]],
            "metrics": {},
        }

    def advance(
        self,
        cell_id: str,
        target_stage: str,
        *,
        actor_id: str,
        metrics: dict[str, Any] | None = None,
    ) -> str:
        if cell_id not in self.cells:
            raise KeyError(f"Unknown cell_id: {cell_id}")
        if target_stage not in self.stages:
            raise ValueError(f"Invalid target stage: {target_stage}")
        entry = self.cells[cell_id]
        expected_idx = int(entry["stage_index"]) + 1
        target_idx = self.stages.index(target_stage)
        if target_idx != expected_idx:
            raise ValueError(f"Cannot jump cell {cell_id} from {entry['stage']} directly to {target_stage}")
        promote_idx = len(self.stages) - 1
        check_idx = promote_idx - 1
        check_stage = self.stages[check_idx]
        promote_stage = self.stages[promote_idx]
        if target_stage == check_stage:
            if actor_id == entry["maker_id"]:
                raise PermissionError(f"Disjoint checker required for {cell_id}: {actor_id}")
            entry["checker_id"] = actor_id
        if target_stage == promote_stage and not entry["checker_id"]:
            raise PermissionError(f"Cannot promote unaudited cell {cell_id}")
        entry["stage_index"] = target_idx
        entry["stage"] = target_stage
        entry["history"].append(target_stage)
        if metrics:
            entry["metrics"].update(metrics)
        return target_stage

    def promote_full_lifecycle(
        self,
        cell_id: str,
        regime_id: str,
        values: dict[str, Any],
        *,
        maker_id: str,
        checker_id: str,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        self.register_cell(cell_id, regime_id, values, maker_id=maker_id)
        promote_idx = len(self.stages) - 1
        check_idx = promote_idx - 1
        checker_stages = {self.stages[check_idx], self.stages[promote_idx]}
        for stage in self.stages[1:]:
            actor = checker_id if stage in checker_stages else maker_id
            stage_metrics = metrics if stage == self.stages[check_idx] else None
            self.advance(cell_id, stage, actor_id=actor, metrics=stage_metrics)
        return dict(self.cells[cell_id])

    def summary(self) -> dict[str, Any]:
        promoted = [cid for cid, rec in self.cells.items() if rec["stage"] == self.stages[-1]]
        promoted_regimes = sorted({self.cells[cid]["regime_id"] for cid in promoted})
        return {
            "registered_cells": len(self.cells),
            "promoted_cells": len(promoted),
            "lifecycle_stage_depth": len(self.stages),
            "terminal_stage": self.stages[-1],
            "disjoint_maker_checker_enforced": all(
                rec["checker_id"] is not None and rec["checker_id"] != rec["maker_id"]
                for rec in self.cells.values()
                if rec["stage"] == self.stages[-1]
            ),
            "promoted_regimes": promoted_regimes,
            "promoted_cell_ids": sorted(promoted),
        }


@dataclass
class NodeGrid:
    """Multi-dimensional Configuration Settings Grid with cross-parameter constraints and auto-expansion."""

    schema_id: str
    generation: int
    parameters: dict[str, NodeParameter]
    constraints: list[dict[str, Any]]
    regimes: dict[str, dict[str, Any]]
    lifecycle_stages: tuple[str, ...] = DEFAULT_LIFECYCLE_STAGES
    raw_spec: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json_file(cls, path: str | Path | None = None) -> NodeGrid:
        resolved = Path(path) if path is not None else default_grid_path()
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        param_map: dict[str, NodeParameter] = {}
        for item in raw["parameters"]:
            p = NodeParameter.from_spec(item)
            param_map[p.axis_id] = p
        stages = tuple(str(s) for s in raw.get("lifecycle_stages", DEFAULT_LIFECYCLE_STAGES))
        return cls(
            schema_id=str(raw["schema_id"]),
            generation=int(raw["generation"]),
            parameters=param_map,
            constraints=list(raw.get("constraints", [])),
            regimes=dict(raw.get("regimes", {})),
            lifecycle_stages=stages,
            raw_spec=raw,
        )

    def validate_cell(self, cell: dict[str, Any]) -> dict[str, Any]:
        validated: dict[str, Any] = {}
        for axis_id, param in self.parameters.items():
            raw_val = cell.get(axis_id, param.default)
            validated[axis_id] = param.validate_value(raw_val)
        for c in self.constraints:
            left = validated[str(c["left_axis"])]
            right = validated[str(c["right_axis"])]
            op = str(c["operator"])
            ok = (
                (left < right if op == "<" else False)
                or (left <= right if op == "<=" else False)
                or (left > right if op == ">" else False)
                or (left >= right if op == ">=" else False)
                or (left == right if op == "==" else False)
            )
            if not ok:
                raise ValueError(f"Cross-parameter constraint failed: {c['constraint_id']} ({left} {op} {right})")
        return validated

    def resolve_regime_cell(self, regime_id: str) -> tuple[str, dict[str, Any]]:
        if regime_id not in self.regimes:
            raise KeyError(f"Unknown regime_id: {regime_id}")
        cell = self.validate_cell(self.regimes[regime_id]["cell"])
        short_len = int(self.raw_spec["numerical_constants"]["short_id_chars"])
        crc_val = zlib.crc32(json.dumps({"regime": regime_id, "cell": cell}, sort_keys=True).encode("utf-8"))
        digest = f"{crc_val:0{short_len}x}"[:short_len]
        return f"{regime_id}_{digest}", cell

    def auto_expand_saturated_boundaries(self, promoted_cells: list[dict[str, Any]]) -> tuple[NodeGrid, list[str]]:
        expanded_axes: list[str] = []
        next_params: dict[str, NodeParameter] = {}
        for axis_id, param in self.parameters.items():
            saturated = any(param.bounds.is_upper_saturated(cell.get(axis_id, param.default)) for cell in promoted_cells)
            if saturated:
                expanded_param = param.expand_upper_boundary()
                if expanded_param != param:
                    expanded_axes.append(axis_id)
                next_params[axis_id] = expanded_param
            else:
                next_params[axis_id] = param
        next_gen = self.generation + (1 if expanded_axes else 0)
        next_grid = NodeGrid(
            schema_id=self.schema_id,
            generation=next_gen,
            parameters=next_params,
            constraints=list(self.constraints),
            regimes=dict(self.regimes),
            lifecycle_stages=self.lifecycle_stages,
            raw_spec=dict(self.raw_spec),
        )
        return next_grid, expanded_axes


class ZeroHardcodedValuesAstGate:
    """Mechanical AST gate rejecting magic numeric literals outside {0, 1, -1, 0.0, 1.0}."""

    def __init__(self, allowed_literals: frozenset[int | float] = ALLOWED_NUMERIC_IDENTITIES) -> None:
        self.allowed_literals = allowed_literals

    def scan_python_file(self, path: str | Path) -> list[dict[str, Any]]:
        file_path = Path(path)
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
        violations: list[dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                if node.value not in self.allowed_literals:
                    violations.append(
                        {
                            "file": file_path.name,
                            "line": int(getattr(node, "lineno", 0)),
                            "value": node.value,
                        }
                    )
        return violations

    def assert_clean(self, paths: list[str | Path]) -> dict[str, Any]:
        all_violations: list[dict[str, Any]] = []
        for p in paths:
            all_violations.extend(self.scan_python_file(p))
        if all_violations:
            raise AssertionError(f"ZeroHardcodedValuesAstGate failed: {all_violations}")
        return {"checked_files": len(paths), "violations": 0}


class DomainAgnosticNamingGate:
    """Mechanical naming gate across filenames, JSON schema IDs, and Python AST definitions."""

    def __init__(self, forbidden_pattern: re.Pattern[str] = FORBIDDEN_DOMAIN_PATTERN) -> None:
        self.forbidden_pattern = forbidden_pattern

    def scan_workspace(self, workspace_root: str | Path) -> list[dict[str, Any]]:
        root = Path(workspace_root)
        violations: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*")):
            if any(part in {".git", "__pycache__", ".pytest_cache"} or part.endswith(".egg-info") for part in path.parts):
                continue
            rel = str(path.relative_to(root))
            if self.forbidden_pattern.search(rel):
                violations.append({"kind": "path", "target": rel})
            if not path.is_file():
                continue
            if path.suffix == ".json":
                try:
                    doc = json.loads(path.read_text(encoding="utf-8"))
                except ValueError:
                    continue
                if isinstance(doc, dict):
                    for k in ("schema_id", "grid_id", "$id"):
                        if k in doc and self.forbidden_pattern.search(str(doc[k])):
                            violations.append({"kind": "json_schema_id", "target": f"{rel}:{doc[k]}"})
            elif path.suffix == ".py":
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        if self.forbidden_pattern.search(node.name):
                            violations.append({"kind": "ast_symbol", "target": f"{rel}:{node.name}"})
        return violations

    def assert_clean(self, workspace_root: str | Path) -> dict[str, Any]:
        violations = self.scan_workspace(workspace_root)
        if violations:
            raise AssertionError(f"DomainAgnosticNamingGate failed: {violations}")
        return {"violations": 0}
