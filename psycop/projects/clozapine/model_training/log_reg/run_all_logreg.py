import subprocess
from pathlib import Path


def run_logreg_models(base_dir: str):
    base_path = Path(base_dir) / "log_reg"

    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {base_path}")

    # Recursively find all hyperparameter.py scripts
    script_paths = list(base_path.glob("lookahead_*/**/hyperparam.py"))

    if not script_paths:
        print("⚠️ No hyperparam.py scripts found.")
        return

    for script_path in script_paths:
        model_dir = script_path.parent
        print(f"🚀 Running logreg model: {model_dir.relative_to(base_path)}")

        try:
            subprocess.run(["python", str(script_path)], check=True)
            print(f"Completed: {model_dir.name}")
        except subprocess.CalledProcessError as e:
            print(f"Failed: {model_dir.name}\n{e}")


if __name__ == "__main__":
    run_logreg_models("E:/ERPERF/psycop-common-1/psycop/projects/clozapine/model_training")
