from psycop.common.cross_experiments.project_getters.cvd_getter import CVDGetter
from psycop.common.cross_experiments.project_getters.ect_getter import ECTGetter
from psycop.common.cross_experiments.project_getters.restraint_getter import RestraintGetter

def eval_df_getters():
    cvd_eval = CVDGetter().get_eval_df()
    ect_eval = ECTGetter().get_eval_df()
    restraint_eval = RestraintGetter().get_eval_df()

    pass

def desired_ppr_getter():
    cfg = CVDGetter.get_cfg()

    pass

if __name__ == "__main__":
    eval_df_getters()
    desired_ppr_getter()
