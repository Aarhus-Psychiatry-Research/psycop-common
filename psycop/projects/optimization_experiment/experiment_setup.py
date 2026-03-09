from psycop.common.cross_experiments.project_getters.t2d_getter import T2DGetter


def get_cfg():
    pass


def overwrite_metric():
    """overwrite binary_auroc metric in cfg"""
    pass


def train_model():
    pass


def save_relevant_metrics():
    pass


if __name__ == "__main__":

    getter = T2DGetter()
    print(getter.get_cfg())