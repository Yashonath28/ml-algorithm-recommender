class MLAdvisor:

    def suggest_metrics(self, problem_type: str, is_imbalanced: bool):
        if problem_type == "classification":
            if is_imbalanced:
                return ["F1-score", "Precision", "Recall", "ROC-AUC"]
            return ["Accuracy", "F1-score", "ROC-AUC"]

        return ["RMSE", "MAE", "R2 Score"]

    def suggest_preprocessing(
        self,
        has_missing: bool,
        has_categorical: bool
    ):
        steps = ["Train-Test Split"]

        if has_missing:
            steps.append("Handle missing values (mean/median/imputation)")

        if has_categorical:
            steps.append("Encode categorical features")

        steps.append("Feature Scaling (StandardScaler / MinMaxScaler)")

        return steps
