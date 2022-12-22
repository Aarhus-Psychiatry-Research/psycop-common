from psycop_model_training.config_schemas.basemodel import BaseModel


class TrainConfSchema(BaseModel):
    """Training configuration."""

    n_splits: int  # ? How do we handle whether to use crossvalidation or train/val splitting?
    n_trials_per_lookahead: int
    n_active_trainers: int  # Number of lookahead windows to train for at once
    n_jobs_per_trainer: int  # Number of jobs to run in parallel for each lookahead window
