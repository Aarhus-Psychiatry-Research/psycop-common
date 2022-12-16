from typing import Optional

from psycop_model_training.utils.config_schemas import BaseModel


class TrainConfSchema(BaseModel):
    """Training configuration."""

    n_splits: int  # ? How do we handle whether to use crossvalidation or train/val splitting?
    n_trials_per_lookahead: int
    n_active_trainers: int  # Number of lookahead windows to train for at once
    n_jobs_per_trainer: int  # Number of jobs to run in parallel for each lookahead window
    random_delay_per_job_seconds: Optional[
        int
    ] = None  # Add random delay based on cfg.train.random_delay_per_job to avoid
    # each job needing the same resources (GPU, disk, network) at the same time
