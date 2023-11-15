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
    
    # TODO: https://github.com/Aarhus-Psychiatry-Research/psycop-common/issues/440 Move the populate registry function to the baseline registry, to make it easier to add new imports to the function
    suggesters = catalogue.create("psycop", "suggester")
