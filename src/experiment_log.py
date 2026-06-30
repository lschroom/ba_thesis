import functools
import inspect
import json
import pathlib
import sqlite3

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "experiments" / "eval_results.db"
DATA_PARAM_NAMES = {
    "y_true",
    "y_pred",
    "mask",
    "masks",
    "arrays",
    "paths",
    "path",
    "pr_path",
    "data_root",
    "db_path",
    "output_dir",
    "result_root",
}
METRIC_COLUMNS = (
    "pixel_accuracy",
    "precision",
    "recall",
    "dice_f1",
    "iou",
    "boundary_iou",
    "quantity_disagreement",
    "allocation_disagreement",
)


def init_db(db_path=None):
    db_path = pathlib.Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_logs (
                creation_tstamp TEXT PRIMARY KEY,
                gen_function TEXT,
                gen_params TEXT,
                pixel_accuracy REAL,
                precision TEXT,
                recall TEXT,
                dice_f1 TEXT,
                iou TEXT,
                boundary_iou TEXT,
                quantity_disagreement REAL,
                allocation_disagreement REAL
            )
            """
        )


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _is_loggable_param(name, value):
    if name in DATA_PARAM_NAMES:
        return False
    if isinstance(value, pathlib.Path):
        return False
    if isinstance(value, np.ndarray):
        return False
    return isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
            type(None),
            list,
            tuple,
            dict,
            np.integer,
            np.floating,
        ),
    )


def filter_generation_params(params):
    return {
        name: value
        for name, value in params.items()
        if _is_loggable_param(name, value)
    }


def _creation_tstamp_from_kwargs(kwargs):
    return (
        kwargs.get("creation_tstamp")
    )


def log_generation(creation_tstamp, gen_function, gen_params, db_path=None):
    init_db(db_path)
    db_path = pathlib.Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    filtered_params = filter_generation_params(gen_params)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO experiment_logs (creation_tstamp, gen_function, gen_params)
            VALUES (?, ?, ?)
            """,
            (
                creation_tstamp,
                gen_function,
                json.dumps(filtered_params, default=_json_default, sort_keys=True),
            ),
        )


def log_metrics(creation_tstamp, metrics, db_path=None):
    init_db(db_path)
    db_path = pathlib.Path(db_path) if db_path is not None else DEFAULT_DB_PATH

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE experiment_logs
            SET pixel_accuracy = ?,
                precision = ?,
                recall = ?,
                dice_f1 = ?,
                iou = ?,
                boundary_iou = ?,
                quantity_disagreement = ?,
                allocation_disagreement = ?
            WHERE creation_tstamp = ?
            """,
            (
                metrics.get("pixel_accuracy"),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("dice_f1"),
                metrics.get("iou"),
                metrics.get("boundary_iou"),
                metrics.get("quantity_disagreement"),
                metrics.get("allocation_disagreement"),
                creation_tstamp,
            ),
        )
        if cursor.rowcount == 0:
            raise ValueError(
                f"No experiment log row exists for creation_tstamp '{creation_tstamp}'."
            )


def log_metric_updates(creation_tstamp, metrics, db_path=None):
    init_db(db_path)
    db_path = pathlib.Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    invalid_metrics = set(metrics) - set(METRIC_COLUMNS)
    if invalid_metrics:
        raise ValueError(f"Unknown metric columns: {sorted(invalid_metrics)}")
    if not metrics:
        return

    columns = [column for column in METRIC_COLUMNS if column in metrics]
    insert_columns = ["creation_tstamp", *columns]
    placeholders = ", ".join(["?"] * len(insert_columns))
    update_clause = ", ".join(
        f"{column} = excluded.{column}"
        for column in columns
    )
    values = [creation_tstamp, *[metrics[column] for column in columns]]

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"""
            INSERT INTO experiment_logs ({", ".join(insert_columns)})
            VALUES ({placeholders})
            ON CONFLICT(creation_tstamp) DO UPDATE SET {update_clause}
            """,
            values,
        )


def log_noise_generation(func):
    """Decorator for generation functions that create a new experiment row."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        creation_tstamp = _creation_tstamp_from_kwargs(kwargs)
        if not creation_tstamp:
            raise ValueError("You must provide a 'creation_tstamp' in kwargs.")

        db_path = kwargs.get("db_path")
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        params = dict(bound_args.arguments)
        params.pop("creation_tstamp", None)


        output = func(*args, **kwargs)
        log_generation(creation_tstamp, func.__name__, params, db_path=db_path)
        return output

    return wrapper


def log_evaluation(func):
    """Decorator that updates metrics for an existing experiment row."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        creation_tstamp = _creation_tstamp_from_kwargs(kwargs)
        if not creation_tstamp:
            raise ValueError("You must provide a 'creation_tstamp' in kwargs.")

        db_path = kwargs.get("db_path")
        results = func(*args, **kwargs)

        init_db(db_path)
        log_metrics(
            creation_tstamp,
            results,
            db_path=db_path,
        )

        return results

    return wrapper
