# Supplementary material

## **eFigure 4**: Performance of xgboost with 5 years of lookahead

![## **eFigure 4**: Performance of xgboost with 5 years of lookahead](figures\t2d_main_performance_figure.png)

A) Receiver operating characteristics (ROC) curve. B) Confusion matrix. PPV: Positive predictive value. NPV: Negative predictive value. C) Sensitivity by months from prediction time to event, stratified by desired predicted positive rate (PPR). Note that the numbers do not match those in Table 1, since all prediction times with insufficient lookahead distance have been dropped. D) Distribution of months from the first positive prediction to the patient fulfilling T2D-criteria at a 3% predicted positive rate (PPR).



## **eFigure 5**: Robustness of xgboost with 5 years of lookahead

![## **eFigure 5**: Robustness of xgboost with 5 years of lookahead](figures\t2d_main_robustness.png)

Robustness of the model across a variety of stratifications. Blue line is the area under the receiver operating characteristics curve. Grey bars represent the number of contacts in each group. Error bars are 95%-confidence intervals from 100-fold bootstrap.



## **eTable 3**: Performance of xgboost with 5 years of lookahead by predicted positive rate (PPR). Numbers are physical contacts.

| Positive rate   | True prevalence   | PPV   | NPV   | Sensitivity   | Specificity   | FPR   | FNR   | Accuracy   |   True positives | True negatives   | False positives   | False negatives   | % of all events captured   |   Mean years from first positive to T2D |
|:----------------|:------------------|:------|:------|:--------------|:--------------|:------|:------|:-----------|-----------------:|:-----------------|:------------------|:------------------|:---------------------------|----------------------------------------:|
| 5.0%            | 1.7%              | 15.6% | 99.1% | 47.3%         | 95.7%         | 4.3%  | 52.7% | 94.9%      |              818 | 98,562           | 4,418             | 912               | 45.1%                      |                                     2.6 |
| 4.0%            | 1.7%              | 18.8% | 99.1% | 45.4%         | 96.7%         | 3.3%  | 54.6% | 95.8%      |              786 | 99,574           | 3,406             | 944               | 41.8%                      |                                     2.6 |
| 3.0%            | 1.7%              | 20.5% | 98.9% | 37.3%         | 97.6%         | 2.4%  | 62.7% | 96.6%      |              645 | 100,482          | 2,498             | 1,085             | 35.9%                      |                                     2.6 |
| 2.0%            | 1.7%              | 23.1% | 98.8% | 28.0%         | 98.4%         | 1.6%  | 72.0% | 97.3%      |              485 | 101,363          | 1,617             | 1,245             | 25.5%                      |                                     2.4 |
| 1.0%            | 1.7%              | 29.2% | 98.6% | 17.7%         | 99.3%         | 0.7%  | 82.3% | 97.9%      |              306 | 102,238          | 742               | 1,424             | 16.3%                      |                                     2.5 |

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

            



