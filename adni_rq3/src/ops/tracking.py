"""Local MLflow tracking helpers for reproducible ADNI experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import mlflow
from mlflow.tracking import MlflowClient


def start_run(experiment_name: str, run_name: str, tracking_dir: str = "mlruns"):
    """Start a local MLflow run using the supported SQLite backend.

    ``tracking_dir`` is retained as the public setting for a simple project
    layout. The tracking database is created beside it as ``mlflow.db`` and
    artifacts are stored under the directory itself.
    """
    artifact_dir = Path(tracking_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    db_path = artifact_dir.parent / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name, artifact_location=artifact_dir.as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name, experiment_id=experiment_id)


def _flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        elif isinstance(value, (str, int, float, bool)):
            flat[name[:250]] = value
    return flat


def log_config(config: dict[str, Any]) -> None:
    """Log scalar configuration values as MLflow parameters."""
    for key, value in _flatten(config).items():
        mlflow.log_param(key, value)


def log_metrics(metrics: dict[str, Any], prefix: str = "") -> None:
    """Log finite numeric metrics and ignore confusion-count metadata."""
    clean: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            name = f"{prefix}.{key}" if prefix else str(key)
            clean[name[:250]] = float(value)
    if clean:
        mlflow.log_metrics(clean)


def log_artifact(path: str | Path) -> None:
    """Log a file when it exists; missing optional artifacts are skipped."""
    path = Path(path)
    if path.exists() and path.is_file():
        mlflow.log_artifact(str(path))


def end_run() -> None:
    """Close the active run if one exists."""
    if mlflow.active_run() is not None:
        mlflow.end_run()
