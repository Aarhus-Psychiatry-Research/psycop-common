import catalogue
from confection import registry


class RegistryWithDict(registry):
    def to_dict(self) -> dict[str, catalogue.Registry]:
        ...


class BaselineRegistry(RegistryWithDict):
    loggers = catalogue.create("psycop", "loggers")
    trainers = catalogue.create("psycop", "trainers")
    data = catalogue.create("psycop", "data")

    tasks = catalogue.create("psycop", "tasks")

    preprocessing = catalogue.create(
        "psycop",
        "preprocessing",
    )
    task_pipelines = catalogue.create(
        "psycop",
        "task_pipelines",
    )
    estimator_steps = catalogue.create(
        "psycop",
        "estimator_steps",
    )

    metrics = catalogue.create("psycop", "metrics")

    def to_dict(self) -> dict[str, catalogue.Registry]:
        return {
            attribute_name: getattr(self, attribute_name)
            for attribute_name in dir(self)
            if isinstance(getattr(self, attribute_name), catalogue.Registry)
        }
