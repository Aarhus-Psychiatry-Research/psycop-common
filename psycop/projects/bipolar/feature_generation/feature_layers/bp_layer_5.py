from collections.abc import Sequence

from timeseriesflattener import PredictorGroupSpec
from timeseriesflattener.v1.aggregation_fns import boolean
from timeseriesflattener.v1.feature_specs.group_specs import NamedDataframe

from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.projects.bipolar.feature_generation.feature_layers.bp_feature_layer import (
    BpFeatureLayer,
)
from psycop.projects.bipolar.feature_generation.feature_layers.bp_text_feature_spec import (
    BpTextFeatures,
)
from psycop.projects.bipolar.feature_generation.feature_layers.value_specification import (
    ValueSpecification,
)


class BpLayer5(BpFeatureLayer):
    def get_features(self, lookbehind_days: list[float]) -> Sequence[ValueSpecification]:
        layer = 5

        note_types = ["aktuelt_psykisk"]
        model_name = "tfidf-500"

        tfidf_features = BpTextFeatures().get_feature_specs(
            note_types=note_types, model_name=model_name, lookbehind_days=lookbehind_days
        )

        return tfidf_features
