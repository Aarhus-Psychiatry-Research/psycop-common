import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
    get_best_eval_pipeline,
)
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    ForcedAdmissionInpatientPipelineRun,
)


def decision_curve_analysis(run: ForcedAdmissionInpatientPipelineRun):
    eval_ds = run.pipeline_outputs.get_eval_dataset()

    # Calculate calibration curve
    y_true, y_prob = eval_ds.y, eval_ds.y_hat_probs

    thresholds = np.linspace(0, 1, 100)
    
    n = len(y_true)
    treat_all = np.mean(y_true)
    treat_none = 0
    
    net_benefit_model = []
    net_benefit_all = []
    net_benefit_none = [0] * len(thresholds)
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        nb_model = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        nb_all = treat_all - (1 - treat_all) * (threshold / (1 - threshold))
        
        net_benefit_model.append(nb_model)
        net_benefit_all.append(nb_all)
    
    results = {
        'thresholds': thresholds,
        'net_benefit_model': net_benefit_model,
        'net_benefit_all': net_benefit_all,
        'net_benefit_none': net_benefit_none,
    }
    
    return results


def plot_decision_curve(run: ForcedAdmissionInpatientPipelineRun, results, model_label="Model"):
    """
    Plot the Decision Curve Analysis (DCA).
    
    Parameters:
    - results: dict
        Dictionary containing thresholds and net benefits.
    - model_label: str, optional
        Label for the model in the plot.
    """
    plt.figure(figsize=(8, 6))
    
    plt.plot(results['thresholds'], results['net_benefit_model'], label=model_label, color="#0072B2")
    plt.plot(results['thresholds'], results['net_benefit_all'], label="Treat All", linestyle="--", color="#009E73")
    plt.plot(results['thresholds'], results['net_benefit_none'], label="Treat None", linestyle="--", color="gray")
    
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.legend(loc="best")
    plt.grid(True)
    dca_path = run.paper_outputs.paths.figures / "fa_inpatient_dca.png"
    plt.savefig(dca_path)
    plt.show()


if __name__ == "__main__":
    
    # Perform Decision Curve Analysis
    results = decision_curve_analysis(run=get_best_eval_pipeline())

    # Plot the Decision Curve
    plot_decision_curve(run= get_best_eval_pipeline(), results=results, model_label="Example Model")