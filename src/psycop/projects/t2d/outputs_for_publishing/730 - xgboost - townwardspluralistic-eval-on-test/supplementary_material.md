# Supplementary material

## **eFigure 4**: Performance of xgboost with 2 years of lookahead

![## **eFigure 4**: Performance of xgboost with 2 years of lookahead](figures\t2d_main_performance_figure.png)

A) Receiver operating characteristics (ROC) curve. B) Confusion matrix. PPV: Positive predictive value. NPV: Negative predictive value. C) Sensitivity by months from prediction time to event, stratified by desired predicted positive rate (PPR). Note that the numbers do not match those in Table 1, since all prediction times with insufficient lookahead distance have been dropped. D) Distribution of months from the first positive prediction to the patient fulfilling T2D-criteria at a 3% predicted positive rate (PPR).



## **eFigure 5**: Robustness of xgboost with 2 years of lookahead

![## **eFigure 5**: Robustness of xgboost with 2 years of lookahead](figures\t2d_main_robustness.png)

Robustness of the model across a variety of stratifications. Blue line is the area under the receiver operating characteristics curve. Grey bars represent the number of contacts in each group. Error bars are 95%-confidence intervals from 100-fold bootstrap.



## **eTable 3**: Performance of xgboost with 2 years of lookahead by predicted positive rate (PPR). Numbers are physical contacts.

| Positive rate   | True prevalence   | PPV   | NPV   | Sensitivity   | Specificity   | FPR   | FNR   | Accuracy   |   True positives | True negatives   | False positives   |   False negatives | % of all events captured   |   Mean years from first positive to T2D |
|:----------------|:------------------|:------|:------|:--------------|:--------------|:------|:------|:-----------|-----------------:|:-----------------|:------------------|------------------:|:---------------------------|----------------------------------------:|
| 5.0%            | 0.4%              | 4.1%  | 99.8% | 55.6%         | 95.2%         | 4.8%  | 44.4% | 95.0%      |              385 | 178,444          | 9,025             |               307 | 42.9%                      |                                     1.2 |
| 4.0%            | 0.4%              | 4.8%  | 99.8% | 52.3%         | 96.2%         | 3.8%  | 47.7% | 96.0%      |              362 | 180,303          | 7,166             |               330 | 38.1%                      |                                     1.1 |
| 3.0%            | 0.4%              | 6.0%  | 99.8% | 49.0%         | 97.2%         | 2.8%  | 51.0% | 97.0%      |              339 | 182,158          | 5,311             |               353 | 33.3%                      |                                     1.2 |
| 2.0%            | 0.4%              | 7.7%  | 99.8% | 42.1%         | 98.1%         | 1.9%  | 57.9% | 97.9%      |              291 | 183,996          | 3,473             |               401 | 27.8%                      |                                     1.1 |
| 1.0%            | 0.4%              | 10.8% | 99.7% | 29.5%         | 99.1%         | 0.9%  | 70.5% | 98.8%      |              204 | 185,790          | 1,679             |               488 | 19.8%                      |                                     1.1 |

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

            



