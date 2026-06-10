import ast
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path("E:/shared_resources/forced_admissions_inpatient/models")


def load_cfg(cfg_path: Path) -> Any:
    with Path.open(cfg_path, "rb") as f:
        return pickle.load(f)


def get_parquet_filename(model_family: str) -> str:
    if model_family == "xgboost":
        return "xgboost_180.parquet"
    if model_family in ["logistic", "logistic_regression"]:
        return "logistic-regression_180.parquet"
    raise ValueError(f"Unknown model family: {model_family}")


def find_best_group(model_dir: Path, model_family: str) -> tuple[Path, Path | None, float]:
    parquet_name = get_parquet_filename(model_family)

    best_score = -float("inf")
    best_group = None
    best_parquet_path = None

    pipeline_eval_dir = model_dir / "pipeline_eval"

    for group_dir in pipeline_eval_dir.iterdir():
        if not group_dir.is_dir():
            continue

        parquet_path = group_dir / parquet_name

        if not parquet_path.exists():
            continue

        try:
            df = pd.read_parquet(parquet_path)

            if "roc_auc" not in df.columns:
                continue

            score = df["roc_auc"].max()

            if score > best_score:
                best_score = score
                best_group = group_dir
                best_parquet_path = parquet_path

        except Exception as e:
            print(f"[WARN] failed reading {parquet_path}: {e}")

    if best_group is None:
        raise ValueError(f"No valid group found for {model_family}")

    return best_group, best_parquet_path, best_score


def find_best_run(group_dir: Path, model_family: str) -> Path:
    for run_dir in group_dir.iterdir():
        cfg_path = run_dir / "cfg.pkl"
        if not cfg_path.exists():
            continue

        cfg = load_cfg(cfg_path)

        model_obj = cfg.model

        model_str = ""

        if hasattr(model_obj, "model_dump"):
            model_str = str(model_obj.model_dump()).lower()

        else:
            model_str = str(model_obj).lower()

        if model_family == "xgboost" and "xgboost" in model_str:
            return run_dir

        if model_family == "logistic_regression" and (
            "logistic" in model_str or "elastic" in model_str
        ):
            return run_dir

    raise ValueError(f"No matching run for {model_family}")


def extract_relevant_hyperparams(cfg: Any) -> dict[str, float | int | str | bool | None]:
    raw = cfg.model

    args = raw.args

    if isinstance(args, str):
        args = ast.literal_eval(args)

    ALLOWED_KEYS = {
        "max_depth",
        "learning_rate",
        "n_estimators",
        "gamma",
        "reg_lambda",
        "reg_alpha",
        "subsample",
        "colsample_bytree",
        "grow_policy",
        "C",
        "alpha",
        "penalty",
        "solver",
        "tol",
        "dual",
        "fit_intercept",
        "max_iter",
        "l1_ratio",
    }

    return {k: v for k, v in args.items() if k in ALLOWED_KEYS}


def extract_preprocessing(cfg: Any) -> dict[str, Any]:
    p = cfg.preprocessing

    # post-split part
    post = p.post_split

    feature_selection = post.feature_selection

    return {
        "imputation_method": post.imputation_method,
        "scaling": post.scaling,
        "feature_selection": {
            "name": getattr(feature_selection, "name", None),
            "percentile": feature_selection.params.get("percentile")
            if feature_selection and feature_selection.params
            else None,
        },
        # pre-split info
        "lookbehind_combination": p.pre_split.lookbehind_combination,
    }


def get_best_model(model_name: str, model_family: str) -> dict[str, Any]:
    model_dir = BASE_DIR / model_name

    group_dir, parquet_path, score = find_best_group(model_dir, model_family)

    best_run = find_best_run(group_dir, model_family)

    cfg = load_cfg(best_run / "cfg.pkl")

    hyperparams = extract_relevant_hyperparams(cfg)
    preprocessing = extract_preprocessing(cfg)

    return {
        "model_name": model_name,
        "model_family": model_family,
        "best_group": group_dir.name,
        "best_run": str(best_run),
        "best_parquet": str(parquet_path),
        "roc_auc": float(score),
        "hyperparameters": hyperparams,
        "preprocessing": preprocessing,
    }


if __name__ == "__main__":
    result = get_best_model(
        model_name="full_model_without_text_features",
        model_family="logistic_regression",  # or "xgboost"
    )

    print(json.dumps(result, indent=2))
