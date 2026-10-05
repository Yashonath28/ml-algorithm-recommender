
from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd

from app.services.dataset_analyzer import DatasetAnalyzer
from app.services.algorithm_recommender import AlgorithmRecommender
from app.services.ml_advisor import MLAdvisor
from app.models.response_models import MLAdviceResponse, DatasetSummary
from app.services.baseline_trainer import BaselineTrainer

router = APIRouter()
trainer = BaselineTrainer()


@router.post(
    "/analyze-dataset",
    response_model=MLAdviceResponse
)
async def analyze_dataset(file: UploadFile = File(...)):

    # 1. Validate file type
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )

    # 2. Read CSV safely
    try:
        df = pd.read_csv(file.file)

    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="CSV file is empty or has no header row"
        )

    except pd.errors.ParserError:
        raise HTTPException(
            status_code=400,
            detail="Invalid CSV format"
        )

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="CSV file encoding is not supported"
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="CSV file has no data rows"
        )

    if len(df.columns) == 0:
        raise HTTPException(
            status_code=400,
            detail="CSV file has no columns"
        )

    # 3. Analyze dataset and generate recommendations
    try:
        analyzer = DatasetAnalyzer(df)
        recommender = AlgorithmRecommender()
        advisor = MLAdvisor()

        target_column = analyzer.detect_target_column()

        if target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="Could not identify a valid target column"
            )

        problem_type = analyzer.detect_problem_type(target_column)

        missing_info = analyzer.analyze_missing_values()
        feature_types = analyzer.analyze_feature_types(target_column)
        imbalance = analyzer.analyze_class_imbalance(target_column)

        recommendations = recommender.recommend(
            problem_type=problem_type,
            num_features=(
                len(feature_types["numerical_features"])
                + len(feature_types["categorical_features"])
            ),
            num_samples=df.shape[0],
            has_categorical=(
                len(feature_types["categorical_features"]) > 0
            ),
            is_imbalanced=(
                imbalance["is_imbalanced"] if imbalance else False
            )
        )

        metrics = advisor.suggest_metrics(
            problem_type,
            imbalance["is_imbalanced"] if imbalance else False
        )

        preprocessing = advisor.suggest_preprocessing(
            has_missing=missing_info["has_missing"],
            has_categorical=(
                len(feature_types["categorical_features"]) > 0
            )
        )

    except HTTPException:
        raise

    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not analyze this dataset: {exc}"
        ) from exc

    # 4. Train and compare multiple ML models
    training_results = trainer.train(
        df=df,
        target_column=target_column,
        problem_type=problem_type
    )

    # Handle training or comparison errors
    if (
        not isinstance(training_results, dict)
        or "error" in training_results
    ):
        detail = (
            training_results.get("error", "Model comparison failed")
            if isinstance(training_results, dict)
            else "Model comparison returned an invalid result"
        )

        raise HTTPException(
            status_code=400,
            detail=detail
        )

    # 5. Return complete analysis and model comparison
    return {
        "dataset_summary": DatasetSummary(
            rows=df.shape[0],
            columns=df.shape[1],
            problem_type=problem_type,
            target_column=target_column
        ),
        "recommended_algorithms": recommendations,
        "suggested_metrics": metrics,
        "preprocessing_steps": preprocessing,
        "class_imbalance": imbalance,
        "baseline_results": training_results["baseline_results"],
        "model_comparison": training_results["model_comparison"],
        "best_model": training_results["best_model"],
        "selection_metric": training_results["selection_metric"]
    }