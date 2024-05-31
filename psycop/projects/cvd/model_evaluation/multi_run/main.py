import auroc_by_run_data as abrd
import auroc_by_run_presentation

if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    result = abrd.get(
        runs=[
            abrd.RunSelector(experiment_name="CVD", run_name="CVD layer 1, base"),
            abrd.RunSelector(experiment_name="CVD", run_name="CVD layer 1, (mean, min, mx)"),
            abrd.RunSelector(experiment_name="CVD", run_name="CVD layer 1, lookbehind: 90,365,730"),
            abrd.RunSelector(experiment_name="CVD", run_name="CVD layer 2, base"),
            abrd.RunSelector(experiment_name="CVD", run_name="CVD layer 3, base"),
            abrd.RunSelector(experiment_name="CVD", run_name="CVD layer 4, base"),
            abrd.RunSelector(
                experiment_name="CVD hyperparam tuning, layer 1, xgboost, v2",
                run_name="Layer 1, hparam",
            ),
            abrd.RunSelector(
                experiment_name="CVD hyperparam tuning, layer 2, xgboost, v2",
                run_name="Layer 2, hparam",
            ),
            abrd.RunSelector(
                experiment_name="CVD hyperparam tuning, layer 3, xgboost, v2",
                run_name="Layer 3, hparam",
            ),
            abrd.RunSelector(
                experiment_name="CVD hyperparam tuning, layer 4, xgboost, v2",
                run_name="Layer 4, hparam",
            ),
        ]
    )

    plot = auroc_by_run_presentation.plot(result)
