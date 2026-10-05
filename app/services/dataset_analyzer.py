
import pandas as pd


class DatasetAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def detect_target_column(self):
        common_targets = [
            "target",
            "label",
            "class",
            "outcome",
            "y",
            "result",
        ]

        if self.df.empty and len(self.df.columns) == 0:
            raise ValueError("Dataset has no columns")

        # Detect commonly used target-column names
        for col in self.df.columns:
            if str(col).strip().lower() in common_targets:
                return col

        # Fallback: use the last column
        if len(self.df.columns) == 0:
            raise ValueError("Dataset has no columns")

        return self.df.columns[-1]

    def detect_problem_type(self, target_column: str):
        if target_column not in self.df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found"
            )

        target_series = self.df[target_column].dropna()

        if target_series.empty:
            raise ValueError("No valid target values available")

        dtype = target_series.dtype

        # Text, category, and boolean targets indicate classification
        if (
            pd.api.types.is_object_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(dtype)
        ):
            return "classification"

        # Numeric targets: use a practical heuristic.
        # Continuous numeric values are treated as regression.
        if pd.api.types.is_numeric_dtype(dtype):
            unique_values = target_series.nunique()

            # Integer labels with only two unique values
            # are commonly binary classification labels.
            if (
                pd.api.types.is_integer_dtype(dtype)
                and unique_values == 2
            ):
                return "classification"

            # Other numeric targets default to regression.
            # This avoids misclassifying small numeric datasets
            # such as scores, prices, and measurements.
            return "regression"

        # Fallback for other target types
        return "classification"

    def analyze_missing_values(self):
        missing = self.df.isnull().sum()
        total_missing = int(missing.sum())

        return {
            "has_missing": total_missing > 0,
            "missing_per_column": {
                str(column): int(count)
                for column, count in missing[missing > 0].items()
            },
        }

    def analyze_feature_types(self, target_column: str):
        if target_column not in self.df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found"
            )

        features = self.df.drop(columns=[target_column])

        numerical = features.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical = features.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns.tolist()

        return {
            "numerical_features": numerical,
            "categorical_features": categorical,
        }

    def analyze_class_imbalance(self, target_column: str):
        if target_column not in self.df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found"
            )

        problem_type = self.detect_problem_type(target_column)

        if problem_type != "classification":
            return None

        target_series = self.df[target_column].dropna()

        if target_series.empty:
            raise ValueError("No valid target values available")

        value_counts = target_series.value_counts(
            normalize=True
        )

        distribution = {
            str(label): float(proportion)
            for label, proportion in value_counts.items()
        }

        # Flag a class distribution where any class
        # represents less than 20% of the valid target values.
        is_imbalanced = any(
            proportion < 0.2
            for proportion in distribution.values()
        )

        return {
            "class_distribution": distribution,
            "is_imbalanced": bool(is_imbalanced),
        }