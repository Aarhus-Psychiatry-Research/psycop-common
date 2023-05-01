# PSYCOP model evaluations
Collection of model evaluations for the PSYCOP project.

The repository is structured like:
```
src/psycop_model_evaluation
├── base_charts.py                            # General charts, e.g. a generic "scatter" chart.
├── binary                                    # Evaluations of binary classification
│   ├── global_performance                    # E.g. precision_recall and roc_auc
│   ├── performance_by_true_positive_rate.py  # Performance for given thresholds
│   ├── subgroups                             # Subgroups analyses, e.g. age and sex
│   └── time                                  # Time analyses. 
│                                             # Absolute (how we usually think of time) 
│                                             # Periodic (e.g. day of week, week of year) and 
│                                             # Timedelta (e.g. time from first visit)
│ 
├── feature_importance                        # Feature importance
│   ├── feature_importance_table.py
│   └── sklearn
│       └── feature_importance.py
├── time_to_event                             # Evaluations for time to event
└── utils.py
```