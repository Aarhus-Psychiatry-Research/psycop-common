from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict


class PSYCOPBaseModel(PydanticBaseModel):
    """."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    def __transform_attributes_with_str_to_object(
        self, output_object: Any, input_string: str = "str"
    ):
        for key, value in self.__dict__.items():
            if isinstance(value, str) and value.lower() == input_string.lower():
                self.__dict__[key] = output_object

    def __init__(self, frozen: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.model_config["frozen"] = frozen

        self.__transform_attributes_with_str_to_object(input_string="null", output_object=None)
        self.__transform_attributes_with_str_to_object(input_string="false", output_object=False)
        self.__transform_attributes_with_str_to_object(input_string="true", output_object=True)
