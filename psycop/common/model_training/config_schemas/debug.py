from pydantic import ConfigDict

from psycop.common.global_utils.pydantic_basemodel import PSYCOPBaseModel


class DebugConfSchema(PSYCOPBaseModel):
    """Configuration options for testing and debugging."""

    model_config = ConfigDict(frozen=True)

    assert_outcome_col_matching_lookahead_exists: bool = True
