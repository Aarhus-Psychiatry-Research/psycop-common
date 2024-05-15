import marimo

__generated_with = "0.5.2"
app = marimo.App()


@app.cell
def __():
    from auroc_by_run_data import data, RunSelector
    return RunSelector, data


@app.cell
def __(RunSelector, data):
    result = data(
        runs=[
            RunSelector(experiment_name="baseline_v2_cvd", run_name="Layer 1"),
            RunSelector(
                experiment_name="baseline_v2_cvd",
                run_name="Layer 1 + agg (min, mean, max)",
            ),
            RunSelector(
                experiment_name="baseline_v2_cvd",
                run_name="Layer 1 + lookbehinds (90, 365, 730)",
            ),
            RunSelector(experiment_name="baseline_v2_cvd", run_name="Layer 2"),
            RunSelector(experiment_name="baseline_v2_cvd", run_name="Layer 3"),
            RunSelector(experiment_name="baseline_v2_cvd", run_name="Layer 4"),
        ]
    )
    return result,


@app.cell
def __(result):
    result
    return


@app.cell
def __():
    import auroc_by_run_presentation
    return auroc_by_run_presentation,


@app.cell
def __(auroc_by_run_presentation, result):
    auroc_by_run_presentation.plot(result)
    return


if __name__ == "__main__":
    app.run()
