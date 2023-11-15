import catalogue
from confection import registry


class RegistryWithDict(registry):
    def to_dict(self) -> dict[str, catalogue.Registry]:
        ...


class BaselineRegistry(RegistryWithDict):
    loggers = catalogue.create("psycop_baseline", "loggers")
    trainers = catalogue.create("psycop_baseline", "trainers")
    data = catalogue.create("psycop_baseline", "data")

    tasks = catalogue.create("psycop_baseline", "tasks")

    preprocessing = catalogue.create(
        "psycop_baseline",
        "preprocessing",
    )
    task_pipelines = catalogue.create(
        "psycop_baseline",
        "task_pipelines",
    )
    estimator_steps = catalogue.create(
        "psycop_baseline",
        "estimator_steps",
    )

    metrics = catalogue.create("psycop", "metrics")
    suggesters = catalogue.create("psycop", "suggester")
    
    def to_dict(self) -> dict[str, catalogue.Registry]:
        return {
            attribute_name: getattr(self, attribute_name)
            for attribute_name in dir(self)
            if isinstance(getattr(self, attribute_name), catalogue.Registry)
        }
