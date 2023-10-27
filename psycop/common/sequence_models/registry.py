import catalogue
from confection import registry


class Registry(registry):
    tasks = catalogue.create("psycop", "encoders")
    layers = catalogue.create("psycop", "layers")
    embedders = catalogue.create("psycop", "embedders")

    datasets = catalogue.create("psycop", "datasets")
    loggers = catalogue.create("psycop", "loggers")

    optimizers = catalogue.create("psycop", "optimizers")
    lr_schedulers = catalogue.create("psycop", "lr_schedulers")
    callbacks = catalogue.create("psycop", "callbacks")
