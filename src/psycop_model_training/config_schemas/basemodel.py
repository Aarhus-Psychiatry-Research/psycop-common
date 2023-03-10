from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Extra


class BaseModel(PydanticBaseModel):
    """."""

    class Config:
        """An pydantic basemodel, which doesn't allow attributes that are not
        defined in the class."""

        allow_mutation = False
        arbitrary_types_allowed = True
        extra = Extra.forbid

    def __transform_attributes_with_str_to_object(
        self,
        output_object: Any,
        input_string: str = "str",
    ):
        for key, value in self.__dict__.items():
            if isinstance(value, str) and value.lower() == input_string.lower():
                self.__dict__[key] = output_object

    def __init__(
        self,
        allow_mutation: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.Config.allow_mutation = allow_mutation

        self.__transform_attributes_with_str_to_object(
            input_string="null",
            output_object=None,
        )
        self.__transform_attributes_with_str_to_object(
            input_string="false",
            output_object=False,
        )
        self.__transform_attributes_with_str_to_object(
            input_string="true",
            output_object=True,
        )
