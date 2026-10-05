
from typing import List, Dict, Optional, Any

from pydantic import BaseModel


class AlgorithmRecommendation(BaseModel):
    algorithm: str
    reason: str


class DatasetSummary(BaseModel):
    rows: int
    columns: int
    problem_type: str
    target_column: str


class MLAdviceResponse(BaseModel):
    dataset_summary: DatasetSummary
    recommended_algorithms: List[AlgorithmRecommendation]
    suggested_metrics: List[str]
    preprocessing_steps: List[str]
    class_imbalance: Optional[Dict] = None
    baseline_results: Optional[Dict[str, Any]] = None
    model_comparison: Optional[List[Dict[str, Any]]] = None
    best_model: Optional[str] = None
    selection_metric: Optional[str] = None