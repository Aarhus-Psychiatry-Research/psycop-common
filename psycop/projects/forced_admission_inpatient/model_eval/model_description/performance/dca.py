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

  # Define thresholds excluding 0 and 1 to avoid division by zero
    thresholds = np.linspace(0.01, 0.99, 100)
    
    # Number of samples
    n = len(y_true)

    # Prevalence of positive class
    treat_all = np.mean(y_true)

    # Net benefits for different strategies
    net_benefit_model = []
    net_benefit_all = []
    net_benefit_none = [0] * len(thresholds)  # Net benefit for treating none is always 0
    
    # Iterate over thresholds to calculate net benefits
    for threshold in thresholds:
        # Predicted positive based on threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # True Positives and False Positives
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        # Net benefit calculation for the model
        nb_model = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        nb_all = treat_all - (1 - treat_all) * (threshold / (1 - threshold))
        
        # Append net benefits
        net_benefit_model.append(nb_model)
        net_benefit_all.append(nb_all)
    
    # Package dca in a dictionary
    dca = {
        'thresholds': thresholds,
        'net_benefit_model': net_benefit_model,
        'net_benefit_all': net_benefit_all,
        'net_benefit_none': net_benefit_none,
    }
    
    return dca

def plot_decision_curve(run: ForcedAdmissionInpatientPipelineRun):

    # Perform Decision Curve Analysis
    dca = decision_curve_analysis(run=run)

    plt.figure(figsize=(8, 6))
    
    plt.plot(dca['thresholds'], dca['net_benefit_model'], label="Model", color="#0072B2")
    plt.plot(dca['thresholds'], dca['net_benefit_all'], label="Treat All", linestyle="--", color="#009E73")
    plt.plot(dca['thresholds'], dca['net_benefit_none'], label="Treat None", linestyle="--", color="gray")
    
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.legend(loc="best")
    plt.grid(True)

    # Set axis limits
    plt.xlim(0, 0.25)   # X-axis limit from 0 to 0.2
    plt.ylim(-0.2, 0.15)    # Y-axis limit from -1 to 1 (or adjust upper bound if needed)
    
    # Save and show the plot
    dca_path = run.paper_outputs.paths.figures / "fa_inpatient_dca.png"
    plt.savefig(dca_path)
    plt.show()

if __name__ == "__main__":
    


    # Plot the Decision Curve
    plot_decision_curve(run= get_best_eval_pipeline())