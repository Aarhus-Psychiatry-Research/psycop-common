# Supplementary material

## **eFigure 1**: Performance of xgboost with 1 years of lookahead

![## **eFigure 1**: Performance of xgboost with 1 years of lookahead](365_xgboost_phototonicunelucidated-eval-on-test\figures\t2d_main_performance_figure.png)

A) Receiver operating characteristics (ROC) curve. B) Confusion matrix. PPV: Positive predictive value. NPV: Negative predictive value. C) Sensitivity by months from prediction time to event, stratified by desired predicted positive rate (PPR). Note that the numbers do not match those in Table 1, since all prediction times with insufficient lookahead distance have been dropped. D) Distribution of months from the first positive prediction to the patient fulfilling T2D-criteria at a 3% predicted positive rate (PPR).



## **eFigure 2**: Robustness of xgboost with 1 years of lookahead

![## **eFigure 2**: Robustness of xgboost with 1 years of lookahead](365_xgboost_phototonicunelucidated-eval-on-test\figures\t2d_main_robustness.png)

Robustness of the model across a variety of stratifications. Blue line is the area under the receiver operating characteristics curve. Grey bars represent the number of contacts in each group. Error bars are 95%-confidence intervals from 100-fold bootstrap.



## **eTable 1**: Performance of xgboost with 1 years of lookahead by predicted positive rate (PPR). Numbers are physical contacts.

| Positive rate   | True prevalence   | PPV   | NPV   | Sensitivity   | Specificity   | FPR   | FNR   | Accuracy   |   True positives | True negatives   | False positives   |   False negatives | % of all events captured   |   Mean years from first positive to T2D |
|:----------------|:------------------|:------|:------|:--------------|:--------------|:------|:------|:-----------|-----------------:|:-----------------|:------------------|------------------:|:---------------------------|----------------------------------------:|
| 5.0%            | 0.1%              | 1.2%  | 99.9% | 50.8%         | 95.1%         | 4.9%  | 49.2% | 95.0%      |              127 | 200,254          | 10,426            |               123 | 57.1%                      |                                     0.6 |
| 4.0%            | 0.1%              | 1.4%  | 99.9% | 47.2%         | 96.1%         | 3.9%  | 52.8% | 96.0%      |              118 | 202,360          | 8,320             |               132 | 48.1%                      |                                     0.6 |
| 3.0%            | 0.1%              | 1.8%  | 99.9% | 45.2%         | 97.0%         | 3.0%  | 54.8% | 97.0%      |              113 | 204,464          | 6,216             |               137 | 44.2%                      |                                     0.6 |
| 2.0%            | 0.1%              | 2.5%  | 99.9% | 42.8%         | 98.0%         | 2.0%  | 57.2% | 98.0%      |              107 | 206,566          | 4,114             |               143 | 35.1%                      |                                     0.6 |
| 1.0%            | 0.1%              | 3.7%  | 99.9% | 31.2%         | 99.0%         | 1.0%  | 68.8% | 99.0%      |               78 | 208,648          | 2,032             |               172 | 29.9%                      |                                     0.6 |

**Predicted positive**: The proportion of contacts predicted positive by the model. Since the model outputs a predicted probability, this is a threshold set by us.
**True prevalence**: The proportion of contacts that qualified for type 2 diabetes within the lookahead window.
**PPV**: Positive predictive value.
**NPV**: Negative predictive value.
**FPR**: False positive rate.
**FNR**: False negative rate.
**TP**: True positives.
**TN**: True negatives.
**FP**: False positives.
**FN**: False negatives.
**Mean warning days**: For all patients with at least one true positive, the number of days from their first positive prediction to their diagnosis.
            



