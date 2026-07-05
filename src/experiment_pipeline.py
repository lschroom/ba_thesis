"""Reproducible orchestration for the label-noise experiments.

The scientific noise and metric functions live in ``noise_generator`` and
``calculate_metrics``.  This module only plans, validates, resumes, and records
their execution.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import pathlib
import random
import sqlite3
import subprocess
import sys
import tomllib
from collections import defaultdict
from typing import Any

import numpy as np

import noise_generator
from calculate_metrics import (
    METRIC_COLUMNS,
    calculate_missing_metrics,
    load_labels,
    normalize_metrics_for_db,
)
from experiment_log import init_db, log_metric_updates


PIPELINE_VERSION = 1
REQUIRED_METHODS = (
    "random_pixels",
    "swap_classes",
    "zoom",
    "shift_scene",
    "shift_objects",
    "boundary_morphology",
    "add_omission",
)
DB_COLUMNS = (
    "creation_tstamp",
    "gen_function",
    "gen_params",
    *METRIC_COLUMNS,
    "pixels_altered",
)


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    path: pathlib.Path
    root: pathlib.Path
    raw: dict[str, Any]

    def resolve(self, value: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(value)
        return path if path.is_absolute() else (self.root / path).resolve()

    @property
    def data_root(self) -> pathlib.Path:
        return self.resolve(self.raw["dataset"]["data_root"])

    @property
    def actual_db(self) -> pathlib.Path:
        return self.resolve(self.raw["outputs"]["actual_db"])

    @property
    def perfect_db(self) -> pathlib.Path:
        return self.resolve(self.raw["outputs"]["perfect_db"])

    @property
    def manifest_path(self) -> pathlib.Path:
        return self.resolve(self.raw["outputs"]["manifest"])

    @property
    def noise_root(self) -> pathlib.Path:
        return self.resolve(self.raw["outputs"]["noise_dir"])


@dataclasses.dataclass(frozen=True)
class ExperimentPlanItem:
    method: str
    params: dict[str, Any]
    repetition: int
    seed: int
    fingerprint: str


@dataclasses.dataclass(frozen=True)
class ValidationReport:
    reference_files: int
    prediction_files: int
    image_shape: tuple[int, ...]
    inference_required: bool


@dataclasses.dataclass(frozen=True)
class GenerationSummary:
    planned: int
    adopted: int
    generated: int
    resumed: int
    manifest: pathlib.Path
    actual_work_db: pathlib.Path
    perfect_work_db: pathlib.Path
    backups: tuple[pathlib.Path, ...]


@dataclasses.dataclass(frozen=True)
class EvaluationSummary:
    planned: int
    actual_evaluated: int
    perfect_evaluated: int
    actual_db: pathlib.Path
    perfect_db: pathlib.Path
    manifest: pathlib.Path


@dataclasses.dataclass(frozen=True)
class RunSummary:
    generation: GenerationSummary
    evaluation: EvaluationSummary


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"Experiment parameters must be finite, got {value!r}")
        return number
    raise TypeError(f"Unsupported experiment value: {value!r}")


def _json(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"))


def _config_hash(config: ExperimentConfig) -> str:
    return hashlib.sha256(_json(config.raw).encode()).hexdigest()


def load_experiment_config(path: str | pathlib.Path) -> ExperimentConfig:
    """Load and structurally validate an experiment TOML file."""
    path = pathlib.Path(path).resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    required_sections = {"project", "dataset", "outputs", "run", "inference", "noise"}
    missing = required_sections - set(raw)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    root_value = pathlib.Path(raw["project"].get("root", "."))
    root = root_value if root_value.is_absolute() else path.parent / root_value
    config = ExperimentConfig(path=path, root=root.resolve(), raw=raw)

    repetitions = raw["run"].get("repetitions")
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("run.repetitions must be a positive integer")
    if not isinstance(raw["run"].get("global_seed"), int):
        raise ValueError("run.global_seed must be an integer")

    build_experiment_plan(config)
    return config


def build_experiment_plan(config: ExperimentConfig) -> list[ExperimentPlanItem]:
    """Expand explicit severity values into deterministic repetitions."""
    entries = config.raw["noise"]
    methods = [entry.get("method") for entry in entries]
    if sorted(methods) != sorted(REQUIRED_METHODS):
        raise ValueError(
            "The configuration must define each canonical noise method exactly once: "
            f"{list(REQUIRED_METHODS)}"
        )

    repetitions = config.raw["run"]["repetitions"]
    global_seed = config.raw["run"]["global_seed"]
    plan: list[ExperimentPlanItem] = []
    fingerprints: set[str] = set()

    for entry in entries:
        method = entry["method"]
        severity_parameter = entry.get("severity_parameter")
        values = entry.get("values")
        if not isinstance(severity_parameter, str) or not severity_parameter:
            raise ValueError(f"{method}: severity_parameter must be a non-empty string")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{method}: values must be a non-empty list")
        if len({_json(value) for value in values}) != len(values):
            raise ValueError(f"{method}: duplicate severity values are not allowed")

        base_params = dict(entry.get("params", {}))
        if severity_parameter in base_params:
            raise ValueError(
                f"{method}: {severity_parameter!r} must be supplied through values, not params"
            )

        for value in values:
            params = {**base_params, severity_parameter: value}
            _canonical(params)
            for repetition in range(repetitions):
                identity = {
                    "method": method,
                    "params": params,
                    "repetition": repetition,
                }
                digest = hashlib.sha256(_json(identity).encode()).hexdigest()
                fingerprint = digest[:20]
                if fingerprint in fingerprints:
                    raise ValueError(f"Duplicate experiment fingerprint: {fingerprint}")
                fingerprints.add(fingerprint)
                seed_bytes = hashlib.sha256(
                    f"{global_seed}:{digest}".encode()
                ).digest()[:8]
                seed = int.from_bytes(seed_bytes, "big") % (2**32)
                plan.append(
                    ExperimentPlanItem(
                        method=method,
                        params=params,
                        repetition=repetition,
                        seed=seed,
                        fingerprint=fingerprint,
                    )
                )
    return plan


def _file_map(directory: pathlib.Path, pattern: str) -> dict[str, pathlib.Path]:
    return {
        path.stem: path
        for path in sorted(directory.glob(pattern), key=lambda item: item.name)
    }


def _dataset_paths(config: ExperimentConfig) -> tuple[dict[str, pathlib.Path], dict[str, pathlib.Path]]:
    dataset = config.raw["dataset"]
    references = _file_map(
        config.data_root / dataset["labels_dir"], dataset["labels_pattern"]
    )
    predictions = _file_map(
        config.data_root / dataset["predictions_dir"], dataset["predictions_pattern"]
    )
    return references, predictions


def validate_experiment_inputs(config: ExperimentConfig) -> ValidationReport:
    """Validate all immutable inputs before generation starts."""
    dataset = config.raw["dataset"]
    if config.noise_root != (config.data_root / "noise").resolve():
        raise ValueError(
            "outputs.noise_dir must point to <dataset.data_root>/noise because "
            "the preserved add_noise API owns that directory convention"
        )
    references, predictions = _dataset_paths(config)
    expected = dataset["expected_images"]
    if len(references) != expected:
        raise ValueError(f"Expected {expected} reference masks, found {len(references)}")
    if not references:
        raise ValueError("No reference masks were found")

    inference_enabled = bool(config.raw["inference"].get("enabled", False))
    checkpoint = config.resolve(config.raw["inference"]["checkpoint"])
    if inference_enabled and not checkpoint.is_file():
        raise FileNotFoundError(f"Inference checkpoint not found: {checkpoint}")
    if not inference_enabled and references.keys() != predictions.keys():
        missing = sorted(references.keys() - predictions.keys())
        extra = sorted(predictions.keys() - references.keys())
        raise ValueError(
            f"Prediction/reference mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )

    first_reference = load_labels(next(iter(references.values())))
    for stem, path in references.items():
        if load_labels(path).shape != first_reference.shape:
            raise ValueError(f"Reference shape mismatch: {path}")
    if not inference_enabled:
        for stem, path in predictions.items():
            if load_labels(path).shape != first_reference.shape:
                raise ValueError(f"Prediction shape mismatch: {path}")

    return ValidationReport(
        reference_files=len(references),
        prediction_files=len(predictions),
        image_shape=tuple(first_reference.shape),
        inference_required=inference_enabled,
    )


def _run_inference_if_enabled(config: ExperimentConfig) -> None:
    inference = config.raw["inference"]
    if not inference.get("enabled", False):
        return
    dataset = config.raw["dataset"]
    result_dir = config.data_root / inference["results_dir"]
    result_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(config.root / "external/oem_sar/Semantic_Segmentation/test.py"),
        "--data_root",
        str(config.data_root / "test" / "sar_images"),
        "--pretrained_model",
        str(config.resolve(inference["checkpoint"])),
        "--save_results",
        str(result_dir),
    ]
    subprocess.run(command, cwd=config.root, check=True)
    references, predictions = _dataset_paths(config)
    if references.keys() != predictions.keys():
        raise ValueError("Inference completed, but prediction filenames do not match references")


def _load_manifest(config: ExperimentConfig) -> dict[str, Any]:
    if not config.manifest_path.exists():
        return {
            "version": PIPELINE_VERSION,
            "config_hash": _config_hash(config),
            "status": "new",
            "experiments": {},
            "backups": [],
        }
    if not config.raw["run"].get("resume", True):
        raise FileExistsError(
            f"A manifest already exists and run.resume is false: {config.manifest_path}"
        )
    manifest = json.loads(config.manifest_path.read_text())
    if manifest.get("version") != PIPELINE_VERSION:
        raise ValueError("Unsupported experiment manifest version")
    if manifest.get("config_hash") != _config_hash(config):
        raise ValueError(
            "The experiment configuration changed. Archive the existing manifest "
            "before starting a different canonical run."
        )
    return manifest


def _save_manifest(config: ExperimentConfig, manifest: dict[str, Any]) -> None:
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.manifest_path.with_suffix(config.manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(config.manifest_path)


def _working_path(path: pathlib.Path) -> pathlib.Path:
    return path.with_name(f"{path.stem}.building{path.suffix}")


def _sqlite_backup(source: pathlib.Path, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as source_conn, sqlite3.connect(destination) as dest_conn:
        source_conn.backup(dest_conn)


def _archive_active_databases(config: ExperimentConfig, manifest: dict[str, Any]) -> list[pathlib.Path]:
    if manifest.get("backups"):
        return [pathlib.Path(path) for path in manifest["backups"]]
    backup_root = config.resolve(config.raw["outputs"]["backup_dir"])
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backups = []
    for source in (config.actual_db, config.perfect_db):
        if source.exists():
            destination = backup_root / f"{source.stem}_{timestamp}{source.suffix}"
            _sqlite_backup(source, destination)
            backups.append(destination)
    manifest["backups"] = [str(path) for path in backups]
    _save_manifest(config, manifest)
    return backups


def _read_rows(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return {
            row["creation_tstamp"]: dict(row)
            for row in conn.execute("SELECT * FROM experiment_logs")
        }


def _write_row(path: pathlib.Path, row: dict[str, Any]) -> None:
    values = [row.get(column) for column in DB_COLUMNS]
    with sqlite3.connect(path) as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO experiment_logs ({', '.join(DB_COLUMNS)}) "
            f"VALUES ({', '.join(['?'] * len(DB_COLUMNS))})",
            values,
        )


def _generator_maps(num_classes: int) -> tuple[dict[str, str], dict[str, str]]:
    method_to_generator = {
        method: function.__name__
        for method, function in noise_generator.noise_methods(num_classes).items()
    }
    return method_to_generator, {value: key for key, value in method_to_generator.items()}


def _artifact_info(
    directory: pathlib.Path,
    references: dict[str, pathlib.Path],
) -> dict[str, Any] | None:
    generated = _file_map(directory, "*.png")
    if generated.keys() != references.keys():
        return None
    digest = hashlib.sha256()
    for stem in sorted(generated):
        generated_array = load_labels(generated[stem])
        if generated_array.shape != load_labels(references[stem]).shape:
            return None
        digest.update(stem.encode())
        with generated[stem].open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return {
        "mask_dir": str(directory),
        "file_count": len(generated),
        "checksum": digest.hexdigest(),
    }


def _noise_directory(config: ExperimentConfig, method: str, timestamp: str) -> pathlib.Path:
    return config.noise_root / method / f"{method}_{timestamp}"


def _params_key(params: dict[str, Any] | str) -> str:
    parsed = json.loads(params) if isinstance(params, str) else params
    return _json(parsed)


def _adopt_legacy(
    config: ExperimentConfig,
    plan: list[ExperimentPlanItem],
    manifest: dict[str, Any],
    actual_work: pathlib.Path,
    perfect_work: pathlib.Path,
    references: dict[str, pathlib.Path],
) -> int:
    if not config.raw["run"].get("adopt_legacy", True):
        return 0
    if manifest.get("legacy_adoption_complete"):
        return 0
    actual_rows = _read_rows(config.actual_db)
    perfect_rows = _read_rows(config.perfect_db)
    num_classes = config.raw["dataset"]["num_classes"]
    method_to_generator, generator_to_method = _generator_maps(num_classes)
    candidates: dict[tuple[str, str], list[str]] = defaultdict(list)
    for timestamp, row in actual_rows.items():
        perfect_row = perfect_rows.get(timestamp)
        method = generator_to_method.get(row.get("gen_function"))
        if method is None or perfect_row is None:
            continue
        if (
            perfect_row.get("gen_function") != row.get("gen_function")
            or _params_key(perfect_row.get("gen_params")) != _params_key(row.get("gen_params"))
        ):
            continue
        candidates[(method, _params_key(row["gen_params"]))].append(timestamp)
    for timestamps in candidates.values():
        timestamps.sort()

    adopted = 0
    for item in plan:
        if item.fingerprint in manifest["experiments"]:
            continue
        key = (item.method, _params_key(item.params))
        while candidates[key]:
            timestamp = candidates[key].pop(0)
            directory = _noise_directory(config, item.method, timestamp)
            artifact = _artifact_info(directory, references)
            if artifact is None:
                continue
            _write_row(actual_work, actual_rows[timestamp])
            _write_row(perfect_work, perfect_rows[timestamp])
            manifest["experiments"][item.fingerprint] = {
                "method": item.method,
                "params": item.params,
                "repetition": item.repetition,
                "seed": None,
                "source": "legacy",
                "timestamp": timestamp,
                **artifact,
            }
            adopted += 1
            break
    manifest["legacy_adoption_complete"] = True
    _save_manifest(config, manifest)
    return adopted


def _delete_work_row(path: pathlib.Path, timestamp: str) -> None:
    if not path.exists():
        return
    with sqlite3.connect(path) as conn:
        conn.execute(
            "DELETE FROM experiment_logs WHERE creation_tstamp = ?", (timestamp,)
        )


def _prune_work_rows(
    actual_work: pathlib.Path,
    perfect_work: pathlib.Path,
    manifest: dict[str, Any],
) -> None:
    allowed = {
        entry["timestamp"]
        for entry in manifest["experiments"].values()
        if entry.get("timestamp")
    }
    for path in (actual_work, perfect_work):
        if not path.exists():
            continue
        with sqlite3.connect(path) as conn:
            rows = conn.execute("SELECT creation_tstamp FROM experiment_logs").fetchall()
            conn.executemany(
                "DELETE FROM experiment_logs WHERE creation_tstamp = ?",
                [(timestamp,) for (timestamp,) in rows if timestamp not in allowed],
            )


def _allocate_timestamp(config: ExperimentConfig, manifest: dict[str, Any]) -> str:
    used = {
        entry["timestamp"]
        for entry in manifest["experiments"].values()
        if entry.get("timestamp")
    }
    used.update(
        entry["timestamp"]
        for entry in manifest.get("superseded", [])
        if entry.get("timestamp")
    )
    candidate = dt.datetime.now().replace(microsecond=0)
    while True:
        timestamp = candidate.strftime("%y%m%d_%H%M%S")
        directory_collision = any(
            config.noise_root.glob(f"*/*_{timestamp}")
        )
        if timestamp not in used and not directory_collision:
            return timestamp
        candidate += dt.timedelta(seconds=1)


def _generate_missing(
    config: ExperimentConfig,
    plan: list[ExperimentPlanItem],
    manifest: dict[str, Any],
    actual_work: pathlib.Path,
    perfect_work: pathlib.Path,
    references: dict[str, pathlib.Path],
) -> int:
    generated_count = 0
    prediction_path = config.data_root / config.raw["dataset"]["predictions_dir"]
    num_classes = config.raw["dataset"]["num_classes"]
    for item in plan:
        existing = manifest["experiments"].get(item.fingerprint)
        if existing:
            directory = pathlib.Path(existing["mask_dir"])
            artifact = _artifact_info(directory, references)
            if artifact is not None and artifact["checksum"] == existing["checksum"]:
                continue
            timestamp = existing.get("timestamp")
            manifest.setdefault("superseded", []).append(
                {
                    **existing,
                    "fingerprint": item.fingerprint,
                    "reason": "missing or changed mask artifact",
                }
            )
            if timestamp:
                _delete_work_row(actual_work, timestamp)
                _delete_work_row(perfect_work, timestamp)
            del manifest["experiments"][item.fingerprint]
            _save_manifest(config, manifest)

        random.seed(item.seed)
        np.random.seed(item.seed)
        timestamp = _allocate_timestamp(config, manifest)
        original_timestamp_factory = noise_generator.creation_tstamp
        noise_generator.creation_tstamp = lambda timestamp=None, value=timestamp: value
        try:
            directories = noise_generator.add_noise(
                prediction_path,
                n_classes=num_classes,
                noise_specs={
                    item.method: {"count": 1, "params": dict(item.params)}
                },
                data_root=config.data_root,
                db_path=actual_work,
            )
        finally:
            noise_generator.creation_tstamp = original_timestamp_factory

        directory = directories[0]
        artifact = _artifact_info(directory, references)
        if artifact is None:
            raise ValueError(f"Generated dataset is incomplete: {directory}")
        manifest["experiments"][item.fingerprint] = {
            "method": item.method,
            "params": item.params,
            "repetition": item.repetition,
            "seed": item.seed,
            "source": "generated",
            "timestamp": timestamp,
            **artifact,
        }
        generated_count += 1
        _save_manifest(config, manifest)
    return generated_count


def _load_stack(paths: dict[str, pathlib.Path]) -> np.ndarray:
    return np.stack([load_labels(paths[stem]) for stem in sorted(paths)])


def _missing_metrics(row: dict[str, Any] | None) -> set[str]:
    if row is None:
        return set(METRIC_COLUMNS)
    return {metric for metric in METRIC_COLUMNS if row.get(metric) is None}


def _ensure_metadata_row(
    source_db: pathlib.Path,
    target_db: pathlib.Path,
    timestamp: str,
) -> None:
    source_row = _read_rows(source_db)[timestamp]
    target_rows = _read_rows(target_db)
    if timestamp not in target_rows:
        _write_row(
            target_db,
            {
                "creation_tstamp": timestamp,
                "gen_function": source_row["gen_function"],
                "gen_params": source_row["gen_params"],
            },
        )


def _evaluate_plan(
    config: ExperimentConfig,
    plan: list[ExperimentPlanItem],
    manifest: dict[str, Any],
    actual_work: pathlib.Path,
    perfect_work: pathlib.Path,
    references: dict[str, pathlib.Path],
    predictions: dict[str, pathlib.Path],
) -> tuple[int, int]:
    clean_labels = _load_stack(references)
    actual_predictions = _load_stack(predictions)
    num_classes = config.raw["dataset"]["num_classes"]
    actual_evaluated = 0
    perfect_evaluated = 0

    for item in plan:
        entry = manifest["experiments"][item.fingerprint]
        timestamp = entry["timestamp"]
        generated_paths = _file_map(pathlib.Path(entry["mask_dir"]), "*.png")
        _ensure_metadata_row(actual_work, perfect_work, timestamp)
        actual_row = _read_rows(actual_work).get(timestamp)
        perfect_row = _read_rows(perfect_work).get(timestamp)
        actual_missing = _missing_metrics(actual_row)
        perfect_missing = _missing_metrics(perfect_row)
        needs_pixels = (
            actual_row.get("pixels_altered") is None
            or perfect_row.get("pixels_altered") is None
        )
        if not actual_missing and not perfect_missing and not needs_pixels:
            entry["status"] = "complete"
            continue

        noisy_labels = _load_stack(generated_paths)
        pixels_altered = int(np.count_nonzero(noisy_labels != clean_labels))
        with sqlite3.connect(actual_work) as conn:
            conn.execute(
                "UPDATE experiment_logs SET pixels_altered = ? WHERE creation_tstamp = ?",
                (pixels_altered, timestamp),
            )
        with sqlite3.connect(perfect_work) as conn:
            conn.execute(
                "UPDATE experiment_logs SET pixels_altered = ? WHERE creation_tstamp = ?",
                (pixels_altered, timestamp),
            )

        if actual_missing:
            values = calculate_missing_metrics(
                noisy_labels, actual_predictions, num_classes, actual_missing
            )
            log_metric_updates(
                timestamp, normalize_metrics_for_db(values), db_path=actual_work
            )
            actual_evaluated += 1
        if perfect_missing:
            values = calculate_missing_metrics(
                noisy_labels, clean_labels, num_classes, perfect_missing
            )
            log_metric_updates(
                timestamp, normalize_metrics_for_db(values), db_path=perfect_work
            )
            perfect_evaluated += 1
        entry["pixels_altered"] = pixels_altered
        entry["status"] = "complete"
        _save_manifest(config, manifest)
    return actual_evaluated, perfect_evaluated


def _validate_database_pair(
    actual_db: pathlib.Path,
    perfect_db: pathlib.Path,
    expected_rows: int,
) -> None:
    actual = _read_rows(actual_db)
    perfect = _read_rows(perfect_db)
    if len(actual) != expected_rows or len(perfect) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, found actual={len(actual)}, perfect={len(perfect)}"
        )
    if actual.keys() != perfect.keys():
        raise ValueError("Actual and perfect databases contain different timestamps")
    for timestamp in actual:
        left, right = actual[timestamp], perfect[timestamp]
        for column in ("gen_function", "gen_params", "pixels_altered"):
            if left[column] != right[column]:
                raise ValueError(f"Paired metadata differs at {timestamp}: {column}")
        missing = [
            column
            for column in (*METRIC_COLUMNS, "pixels_altered")
            if left[column] is None or right[column] is None
        ]
        if missing:
            raise ValueError(f"Incomplete metrics at {timestamp}: {missing}")


def _invalid_manifest_artifacts(
    manifest: dict[str, Any], references: dict[str, pathlib.Path]
) -> list[str]:
    invalid = []
    for fingerprint, entry in manifest["experiments"].items():
        artifact = _artifact_info(pathlib.Path(entry["mask_dir"]), references)
        if artifact is None or artifact["checksum"] != entry.get("checksum"):
            invalid.append(fingerprint)
    return invalid


def generate_experiments(
    config: ExperimentConfig | str | pathlib.Path,
) -> GenerationSummary:
    """Generate or adopt every planned noisy dataset without evaluating it."""
    if not isinstance(config, ExperimentConfig):
        config = load_experiment_config(config)
    plan = build_experiment_plan(config)
    manifest = _load_manifest(config)

    _run_inference_if_enabled(config)
    validation = validate_experiment_inputs(config)
    references, predictions = _dataset_paths(config)

    if manifest.get("status") == "complete":
        invalid_artifacts = _invalid_manifest_artifacts(manifest, references)
        if not invalid_artifacts:
            _validate_database_pair(config.actual_db, config.perfect_db, len(plan))
            return GenerationSummary(
                planned=len(plan),
                adopted=sum(e.get("source") == "legacy" for e in manifest["experiments"].values()),
                generated=sum(e.get("source") == "generated" for e in manifest["experiments"].values()),
                resumed=len(plan),
                manifest=config.manifest_path,
                actual_work_db=config.actual_db,
                perfect_work_db=config.perfect_db,
                backups=tuple(pathlib.Path(path) for path in manifest.get("backups", [])),
            )
        manifest["status"] = "repairing"
        _save_manifest(config, manifest)

    backups = _archive_active_databases(config, manifest)
    actual_work = _working_path(config.actual_db)
    perfect_work = _working_path(config.perfect_db)
    actual_work.parent.mkdir(parents=True, exist_ok=True)
    if not actual_work.exists():
        if manifest.get("status") == "repairing" and config.actual_db.exists():
            _sqlite_backup(config.actual_db, actual_work)
        else:
            init_db(actual_work)
    if not perfect_work.exists():
        if manifest.get("status") == "repairing" and config.perfect_db.exists():
            _sqlite_backup(config.perfect_db, perfect_work)
        else:
            init_db(perfect_work)

    adopted = _adopt_legacy(
        config, plan, manifest, actual_work, perfect_work, references
    )
    _prune_work_rows(actual_work, perfect_work, manifest)
    generated = _generate_missing(
        config, plan, manifest, actual_work, perfect_work, references
    )
    if len(manifest["experiments"]) != len(plan):
        raise ValueError(
            f"Generation produced {len(manifest['experiments'])}/{len(plan)} planned datasets"
        )
    manifest["status"] = "generated"
    manifest["generated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["validation"] = dataclasses.asdict(validation)
    _save_manifest(config, manifest)

    return GenerationSummary(
        planned=len(plan),
        adopted=adopted,
        generated=generated,
        resumed=len(plan) - adopted - generated,
        manifest=config.manifest_path,
        actual_work_db=actual_work,
        perfect_work_db=perfect_work,
        backups=tuple(backups),
    )


def evaluate_experiments(
    config: ExperimentConfig | str | pathlib.Path,
) -> EvaluationSummary:
    """Evaluate previously generated datasets and activate both databases."""
    if not isinstance(config, ExperimentConfig):
        config = load_experiment_config(config)
    plan = build_experiment_plan(config)
    manifest = _load_manifest(config)
    validate_experiment_inputs(config)
    references, predictions = _dataset_paths(config)

    if manifest.get("status") == "complete":
        _validate_database_pair(config.actual_db, config.perfect_db, len(plan))
        return EvaluationSummary(
            planned=len(plan),
            actual_evaluated=0,
            perfect_evaluated=0,
            actual_db=config.actual_db,
            perfect_db=config.perfect_db,
            manifest=config.manifest_path,
        )
    if manifest.get("status") not in {"generated", "evaluating"}:
        raise RuntimeError(
            "Experiment data is not ready. Run generate_experiments(config) first."
        )
    if len(manifest["experiments"]) != len(plan):
        raise RuntimeError(
            "The manifest does not contain every planned dataset. "
            "Run generate_experiments(config) first."
        )
    invalid = _invalid_manifest_artifacts(manifest, references)
    if invalid:
        raise RuntimeError(
            f"{len(invalid)} generated dataset(s) are missing or changed. "
            "Run generate_experiments(config) to repair them before evaluation."
        )

    actual_work = _working_path(config.actual_db)
    perfect_work = _working_path(config.perfect_db)
    if not actual_work.exists() or not perfect_work.exists():
        raise FileNotFoundError(
            "Working databases are missing. Run generate_experiments(config) first."
        )
    manifest["status"] = "evaluating"
    _save_manifest(config, manifest)
    actual_evaluated, perfect_evaluated = _evaluate_plan(
        config,
        plan,
        manifest,
        actual_work,
        perfect_work,
        references,
        predictions,
    )
    _validate_database_pair(actual_work, perfect_work, len(plan))

    os.replace(actual_work, config.actual_db)
    os.replace(perfect_work, config.perfect_db)
    manifest["status"] = "complete"
    manifest["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _save_manifest(config, manifest)

    return EvaluationSummary(
        planned=len(plan),
        actual_evaluated=actual_evaluated,
        perfect_evaluated=perfect_evaluated,
        actual_db=config.actual_db,
        perfect_db=config.perfect_db,
        manifest=config.manifest_path,
    )


def run_experiment(config: ExperimentConfig | str | pathlib.Path) -> RunSummary:
    """Compatibility convenience wrapper for generation followed by evaluation."""
    if not isinstance(config, ExperimentConfig):
        config = load_experiment_config(config)
    generation = generate_experiments(config)
    evaluation = evaluate_experiments(config)
    return RunSummary(generation=generation, evaluation=evaluation)


def database_summary(config: ExperimentConfig | str | pathlib.Path) -> list[dict[str, Any]]:
    """Return per-generator row and completeness counts for notebook display."""
    if not isinstance(config, ExperimentConfig):
        config = load_experiment_config(config)
    output = []
    for predictor, path in (("actual", config.actual_db), ("perfect", config.perfect_db)):
        with sqlite3.connect(path) as conn:
            for generator, rows, complete in conn.execute(
                f"""
                SELECT gen_function,
                       COUNT(*),
                       SUM(CASE WHEN {' AND '.join(f'{column} IS NOT NULL' for column in (*METRIC_COLUMNS, 'pixels_altered'))}
                                THEN 1 ELSE 0 END)
                FROM experiment_logs
                GROUP BY gen_function
                ORDER BY gen_function
                """
            ):
                output.append(
                    {
                        "predictor": predictor,
                        "generator": generator,
                        "rows": rows,
                        "complete_rows": complete,
                    }
                )
    return output
