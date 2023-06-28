from psycop.common.global_utils.pydantic_basemodel import PSYCOPBaseModel


class DebugConfSchema(PSYCOPBaseModel):
    """Configuration options for testing and debugging."""

    class Config:
        """An pydantic basemodel, which doesn't allow attributes that are not
        defined in the class."""

        allow_mutation = False

    assert_outcome_col_matching_lookahead_exists: bool = True
