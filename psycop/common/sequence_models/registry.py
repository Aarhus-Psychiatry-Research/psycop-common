from typing import TypeVar

import catalogue
from confection import registry


class SequenceRegistry(registry):
    tasks = catalogue.create("psycop", "encoders")
    layers = catalogue.create("psycop", "layers")
    embedders = catalogue.create("psycop", "embedders")

    preprocessing = catalogue.create("psycop", "preprocessing")
    datasets = catalogue.create("psycop", "datasets")
    cohorts = catalogue.create("psycop", "cohorts")
    event_loaders = catalogue.create("psycop", "event_loaders")
    loggers = catalogue.create("psycop", "loggers")

    optimizers = catalogue.create("psycop", "optimizers")
    lr_schedulers = catalogue.create("psycop", "lr_schedulers")
    callbacks = catalogue.create("psycop", "callbacks")

    utilities = catalogue.create("psycop", "utilities")


T = TypeVar("T")


@SequenceRegistry.utilities.register("list_creator")
def list_creator(*args: T) -> list[T]:
    return list(args)
