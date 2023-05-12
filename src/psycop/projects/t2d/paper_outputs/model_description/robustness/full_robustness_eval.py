from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_by_age import (
    roc_auc_by_age,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_by_calendar_time import (
    roc_auc_by_calendar_time,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_by_cyclic_time import (
    roc_auc_by_cyclic_time,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_by_n_hba1c import (
    roc_auc_by_n_hba1c,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_by_sex import (
    roc_auc_by_sex,
)
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_by_time_from_first_visit import (
    roc_auc_by_time_from_first_visit,
)

if __name__ == "__main__":
    roc_auc_by_sex()
    roc_auc_by_age()
    roc_auc_by_n_hba1c()
    roc_auc_by_time_from_first_visit()
    roc_auc_by_cyclic_time()
    # roc_auc_by_calendar_time()
