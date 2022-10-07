# pylint: skip-file
from pathlib import Path

import wandb
from wandb.sdk.wandb_run import Run as wandb_run


def log_image_to_wandb(chart_path: Path, chart_name: str, run: wandb_run):
    """Helper to log image to wandb.

    Args:
        chart_path (Path): Path to the image
        chart_name (str): Name of the chart
        run (wandb.run): Wandb run object
    """
    run.log({f"image_{chart_name}": wandb.Image(str(chart_path))})
