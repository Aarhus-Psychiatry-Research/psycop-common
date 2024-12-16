import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_diabetes_hospital
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    data = fetch_diabetes_hospital(as_frame=True)
    X = data.data.copy()
    X = X.drop(columns=["readmitted", "readmit_binary"])
    y = data.target
    X_ohe = pd.get_dummies(X)
    race = X["race"]
    race.value_counts()

    np.random.seed(42)  # set seed for consistent results
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X_ohe, y, race, random_state=123
    )
    classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict_proba(X_test)[:, 1] >= 0.1
    mf = MetricFrame(
        metrics=accuracy_score, y_true=y_test, y_pred=y_pred, sensitive_features=A_test
    )
    mf.overall.item()
    mf.by_group
    mf.difference(method="to_overall")
    mf.ratio(method="to_overall")

    mf = MetricFrame(
        metrics=false_negative_rate, y_true=y_test, y_pred=y_pred, sensitive_features=A_test
    )
    mf.overall.item()

    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "false positive rate": false_positive_rate,
        "false negative rate": false_negative_rate,
        "selection rate": selection_rate,
        "count": count,
    }
    metric_frame = MetricFrame(
        metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=A_test
    )
    metric_frame.by_group.plot.bar(
        subplots=True, layout=[3, 3], legend=False, figsize=[12, 8], title="Show all metrics"
    )

