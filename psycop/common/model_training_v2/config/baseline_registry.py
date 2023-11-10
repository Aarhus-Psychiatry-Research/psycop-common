import catalogue
from confection import registry


class BaselineRegistry(registry):
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
