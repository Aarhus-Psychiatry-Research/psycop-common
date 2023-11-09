import catalogue
from confection import registry


class Registry(registry):
    baseline_tasks = catalogue.create("psycop", "baseline_tasks")
    baseline_loggers = catalogue.create("psycop", "baseline_loggers")

    baseline_preprocessing = catalogue.create(
        "psycop", "baseline_preprocessing"
    )
    baseline_task_pipelines = catalogue.create(
        "psycop", "baseline_task_pipelines"
    )