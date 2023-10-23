"""Model configuration schemas."""
from psycop.common.global_utils.pydantic_basemodel import PSYCOPBaseModel


class ModelConfSchema(PSYCOPBaseModel):
    """Model configuration."""

    name: str  # Model, can currently take xgboost
    require_imputation: bool  # Whether the model requires imputation. (shouldn't this be false?)
    args: dict  # type: ignore
