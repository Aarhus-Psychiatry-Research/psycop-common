import numpy as np
from psycop.model_evaluation.feature_importance.sklearn.plot_feature_importances import (
    plot_feature_importances,
)
from psycop.model_evaluation.utils import TEST_PLOT_PATH


def test_plot_feature_importances():
    n_features = 10
    feature_name = "very long feature name right here yeah actually super long like the feature names"
    feature_names = [feature_name + str(i) for i in range(n_features)]
    # generate 10 random numbers between 0 and 1
    feature_importance = np.random.rand(n_features)

    feature_importance_dict = dict(zip(feature_names, feature_importance))

    plot_feature_importances(
        feature_importance_dict=feature_importance_dict,
        top_n_feature_importances=n_features,
        save_path=TEST_PLOT_PATH / "tmp",
    )
