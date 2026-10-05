class AlgorithmRecommender:

    def recommend(
        self,
        problem_type: str,
        num_features: int,
        num_samples: int,
        has_categorical: bool,
        is_imbalanced: bool
    ):
        recommendations = []

        if problem_type == "classification":
            # Logistic Regression
            if num_samples < 100_000:
                recommendations.append({
                    "algorithm": "Logistic Regression",
                    "reason": "Good baseline for classification, interpretable, works well on small to medium datasets"
                })

            # Random Forest
            recommendations.append({
                "algorithm": "Random Forest Classifier",
                "reason": "Handles non-linearity, robust to noise, works well without heavy feature engineering"
            })

            # XGBoost / Gradient Boosting
            recommendations.append({
                "algorithm": "Gradient Boosting",
                "reason": "High performance on structured/tabular data"
            })

            # Imbalanced datasets
            if is_imbalanced:
                recommendations.append({
                    "algorithm": "XGBoost with class weights / SMOTE",
                    "reason": "Handles class imbalance better"
                })

        elif problem_type == "regression":
            recommendations.append({
                "algorithm": "Linear Regression",
                "reason": "Simple, interpretable baseline"
            })

            recommendations.append({
                "algorithm": "Random Forest Regressor",
                "reason": "Handles non-linear relationships well"
            })

            recommendations.append({
                "algorithm": "Gradient Boosting Regressor",
                "reason": "Strong performance for tabular regression tasks"
            })

        # Categorical-heavy datasets
        if has_categorical:
            recommendations.append({
                "algorithm": "CatBoost",
                "reason": "Handles categorical features natively"
            })

        return recommendations
