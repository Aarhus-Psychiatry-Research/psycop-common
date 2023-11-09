import catalogue
from confection import registry


class BaselineRegistry(registry):
    baseline_loggers = catalogue.create("psycop", "baseline_loggers")
    baseline_trainers = catalogue.create("psycop", "baseline_trainers")
    baseline_data = catalogue.create("psycop", "baseline_data")

    baseline_tasks = catalogue.create("psycop", "baseline_tasks")

    baseline_preprocessing = catalogue.create(
        "psycop", "baseline_preprocessing"
    )
    baseline_task_pipelines = catalogue.create(
        "psycop", "baseline_task_pipelines"
    )
    baseline_estimator_steps = catalogue.create(
        "psycop", "baseline_estimator_steps"
    )

    baseline_metrics = catalogue.create("psycop", "baseline_metrics")