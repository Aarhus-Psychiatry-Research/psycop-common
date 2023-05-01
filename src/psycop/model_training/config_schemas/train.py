from psycop.model_training.config_schemas.basemodel import BaseModel


class TrainConfSchema(BaseModel):
    """Training configuration."""

    # training is done using crossvalidation if splits_for_evaluation is None in DataSchema. If splits are provided for splits_for_evaluation, the trained model is validated on the specified split(s).
    n_trials_per_lookahead: int
    n_active_trainers: int  # Number of lookahead windows to train for at once
    n_jobs_per_trainer: int  # Number of jobs to run in parallel for each lookahead window
