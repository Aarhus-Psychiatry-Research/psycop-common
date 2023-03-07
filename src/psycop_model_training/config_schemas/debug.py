from psycop_model_training.config_schemas.basemodel import BaseModel


class DebugConfSchema(BaseModel):
    """Configuration options for testing and debugging."""

    class Config:
        """An pydantic basemodel, which doesn't allow attributes that are not
        defined in the class."""

        allow_mutation = False

    assert_outcome_col_matching_lookahead_exists: bool = True
