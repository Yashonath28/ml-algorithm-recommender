
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class BaselineTrainer:

    def _build_preprocessor(self, X):
        numerical_features = X.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            exclude="number"
        ).columns.tolist()

        transformers = []

        if numerical_features:
            numerical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            transformers.append((
                "numerical",
                numerical_pipeline,
                numerical_features,
            ))

        if categorical_features:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ])

            transformers.append((
                "categorical",
                categorical_pipeline,
                categorical_features,
            ))

        if not transformers:
            raise ValueError("No usable features available")

        return ColumnTransformer(transformers=transformers)

    def train(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: str,
    ):
        if target_column not in df.columns:
            return {"error": "Target column not found"}

        data = df.dropna(subset=[target_column]).copy()

        if data.empty:
            return {"error": "No valid target values available"}

        X = data.drop(columns=[target_column])
        y = data[target_column]

        X = X.dropna(axis=1, how="all")

        if X.empty:
            return {"error": "No usable features available"}

        numerical_columns = X.select_dtypes(
            include="number"
        ).columns

        X[numerical_columns] = X[numerical_columns].replace(
            [float("inf"), float("-inf")],
            float("nan"),
        )

        # Remove rows whose target is infinite.
        if problem_type == "regression":
            y_numeric = pd.to_numeric(y, errors="coerce")
            valid_target = y_numeric.notna() & y_numeric.map(
                lambda value: abs(value) != float("inf")
            )
            X = X.loc[valid_target]
            y = y_numeric.loc[valid_target]

        if len(X) < 5:
            return {
                "error": "At least 5 valid rows are required for model comparison"
            }

        if problem_type == "classification":
            if y.nunique() < 2:
                return {
                    "error": "Classification requires at least two classes"
                }

            models = [
                ("Logistic Regression", LogisticRegression(max_iter=1000)),
                (
                    "Random Forest Classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                    ),
                ),
                (
                    "Gradient Boosting Classifier",
                    GradientBoostingClassifier(random_state=42),
                ),
            ]

        elif problem_type == "regression":
            if not pd.api.types.is_numeric_dtype(y):
                return {
                    "error": "Regression requires a numeric target"
                }

            models = [
                ("Linear Regression", LinearRegression()),
                (
                    "Random Forest Regressor",
                    RandomForestRegressor(
                        n_estimators=100,
                        random_state=42,
                    ),
                ),
                (
                    "Gradient Boosting Regressor",
                    GradientBoostingRegressor(random_state=42),
                ),
            ]

        else:
            return {"error": "Unsupported problem type"}

        class_counts = y.value_counts() if problem_type == "classification" else None

        if problem_type == "classification" and (
            len(y) < 10 or class_counts.min() < 2
        ):
            return {
                "error": (
                    "Not enough rows per class for a reliable train-test "
                    "comparison. Add more samples for each class."
                )
            }

        stratify = y if problem_type == "classification" else None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=stratify,
            )

            if X_train.empty or X_test.empty:
                return {
                    "error": "Not enough rows to split the dataset"
                }

            comparison = []

            for model_name, model in models:
                try:
                    pipeline = Pipeline([
                        ("preprocessing", self._build_preprocessor(X_train)),
                        ("model", model),
                    ])

                    pipeline.fit(X_train, y_train)
                    predictions = pipeline.predict(X_test)

                    if problem_type == "classification":
                        result = {
                            "model": model_name,
                            "accuracy": float(
                                accuracy_score(y_test, predictions)
                            ),
                            "f1_score": float(
                                f1_score(
                                    y_test,
                                    predictions,
                                    average="weighted",
                                    zero_division=0,
                                )
                            ),
                        }
                    else:
                        rmse = mean_squared_error(
                            y_test,
                            predictions,
                        ) ** 0.5

                        r2 = (
                            float(r2_score(y_test, predictions))
                            if len(y_test) >= 2
                            else None
                        )

                        result = {
                            "model": model_name,
                            "rmse": float(rmse),
                            "mae": float(
                                mean_absolute_error(y_test, predictions)
                            ),
                            "r2_score": r2,
                        }

                    comparison.append(result)

                except (ValueError, TypeError) as exc:
                    comparison.append({
                        "model": model_name,
                        "error": str(exc),
                    })

            successful = [
                result for result in comparison
                if "error" not in result
            ]

            if not successful:
                return {
                    "error": "All model training attempts failed",
                    "model_comparison": comparison,
                }

            if problem_type == "classification":
                best_model = max(
                    successful,
                    key=lambda result: result["f1_score"],
                )
            else:
                best_model = min(
                    successful,
                    key=lambda result: result["rmse"],
                )

            # Keep the existing baseline_results field compatible.
            baseline = next(
                (
                    result for result in successful
                    if result["model"] == (
                        "Logistic Regression"
                        if problem_type == "classification"
                        else "Linear Regression"
                    )
                ),
                successful[0],
            )

            return {
                "baseline_results": baseline,
                "model_comparison": comparison,
                "best_model": best_model["model"],
                "selection_metric": (
                    "weighted_f1_score"
                    if problem_type == "classification"
                    else "rmse"
                ),
            }

        except (ValueError, TypeError) as exc:
            return {"error": f"Model comparison failed: {exc}"}