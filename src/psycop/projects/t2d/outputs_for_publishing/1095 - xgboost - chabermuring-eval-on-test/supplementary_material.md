# Supplementary material

## **eFigure 4**: Performance of xgboost with 3 years of lookahead

![## **eFigure 4**: Performance of xgboost with 3 years of lookahead](figures\t2d_main_performance_figure.png)

A) Receiver operating characteristics (ROC) curve. B) Confusion matrix. PPV: Positive predictive value. NPV: Negative predictive value. C) Sensitivity by months from prediction time to event, stratified by desired predicted positive rate (PPR). Note that the numbers do not match those in Table 1, since all prediction times with insufficient lookahead distance have been dropped. D) Distribution of months from the first positive prediction to the patient fulfilling T2D-criteria at a 3% predicted positive rate (PPR).



## **eFigure 5**: Robustness of xgboost with 3 years of lookahead

![## **eFigure 5**: Robustness of xgboost with 3 years of lookahead](figures\t2d_main_robustness.png)

Robustness of the model across a variety of stratifications. Blue line is the area under the receiver operating characteristics curve. Grey bars represent the number of contacts in each group. Error bars are 95%-confidence intervals from 100-fold bootstrap.



## **eTable 3**: Performance of xgboost with 3 years of lookahead by predicted positive rate (PPR). Numbers are physical contacts.

| Positive rate   | True prevalence   | PPV   | NPV   | Sensitivity   | Specificity   | FPR   | FNR   | Accuracy   |   True positives | True negatives   | False positives   |   False negatives | % of all events captured   |   Mean years from first positive to T2D |
|:----------------|:------------------|:------|:------|:--------------|:--------------|:------|:------|:-----------|-----------------:|:-----------------|:------------------|------------------:|:---------------------------|----------------------------------------:|
| 5.0%            | 0.7%              | 7.2%  | 99.7% | 52.9%         | 95.3%         | 4.7%  | 47.1% | 95.0%      |              588 | 154,072          | 7,551             |               523 | 41.1%                      |                                     1.7 |
| 4.0%            | 0.7%              | 7.7%  | 99.6% | 45.4%         | 96.3%         | 3.7%  | 54.6% | 95.9%      |              504 | 155,614          | 6,009             |               607 | 39.2%                      |                                     1.6 |
| 3.0%            | 0.7%              | 9.8%  | 99.6% | 42.9%         | 97.3%         | 2.7%  | 57.1% | 96.9%      |              477 | 157,218          | 4,405             |               634 | 36.1%                      |                                     1.6 |
| 2.0%            | 0.7%              | 12.3% | 99.6% | 36.0%         | 98.2%         | 1.8%  | 64.0% | 97.8%      |              400 | 158,767          | 2,856             |               711 | 25.3%                      |                                     1.7 |
| 1.0%            | 0.7%              | 15.4% | 99.5% | 22.6%         | 99.1%         | 0.9%  | 77.4% | 98.6%      |              251 | 160,246          | 1,377             |               860 | 18.4%                      |                                     1.6 |

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

            



