# 🧠 ML Algorithm Recommender

An intelligent, API-first Machine Learning advisory system that analyzes a CSV dataset, identifies the machine learning problem type, recommends suitable algorithms, suggests preprocessing steps and evaluation metrics, compares multiple baseline models, and identifies the best-performing model.

The project combines a **FastAPI backend**, **Scikit-learn ML pipeline**, and **React frontend dashboard** to provide an end-to-end dataset analysis experience.

---

## 🚀 Key Features

### 📊 Automatic Dataset Analysis

- Automatically detects the target column
- Identifies the problem type:
  - Classification
  - Regression
- Analyzes missing values
- Identifies numerical and categorical features
- Checks class imbalance for classification datasets
- Handles invalid and unsupported dataset cases

---

### 🤖 Algorithm Recommendation Engine

The system recommends suitable machine learning algorithms based on the detected problem type.

#### Classification

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

#### Regression

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Each recommendation includes a reason explaining why the algorithm is suitable.

---

### 🧪 Metrics & Preprocessing Suggestions

The system provides suitable evaluation metrics based on the detected problem type.

#### Classification Metrics

- Accuracy
- F1-score
- ROC-AUC

#### Regression Metrics

- RMSE
- MAE
- R² Score

The system also suggests preprocessing steps such as:

- Train-Test Split
- Feature Scaling
- Numerical feature preprocessing
- Categorical feature preprocessing

---

### ⚡ Multiple Model Comparison

The system evaluates multiple suitable machine learning models instead of relying on a single baseline model.

#### Classification

- Accuracy
- Weighted F1 Score

#### Regression

- RMSE
- MAE
- R² Score

All evaluated models are compared using the same train-test split.

---

### 🏆 Best Model Selection

The system automatically identifies the best-performing model using the primary selection metric.

- Classification → Weighted F1 Score
- Regression → RMSE

The selected model is displayed clearly in the frontend dashboard.

> Note: Model selection is based on the evaluated test split and should not be considered a guarantee of real-world performance.

---

### 📈 Model Comparison Visualization

The frontend provides an interactive model comparison chart.

- Classification → F1 Score comparison
- Regression → RMSE comparison

A detailed comparison table is displayed below the chart.

---

### 🌐 Professional React Dashboard

The frontend provides:

- CSV file upload
- CSV file validation
- Loading spinner during analysis
- Dataset summary
- Problem type display
- Target column display
- Recommended algorithms
- Suggested metrics
- Preprocessing suggestions
- Class imbalance information
- Model comparison chart
- Model comparison table
- Best model result
- User-friendly error messages

---

# 🛠️ Tech Stack

## Backend

- Python
- FastAPI
- Pandas
- NumPy
- Scikit-learn
- Pydantic
- Uvicorn

## Frontend

- React
- Vite
- Axios
- Recharts
- HTML
- CSS

## Development Tools

- Visual Studio Code
- Git
- GitHub

---

# 📁 Project Structure

```text
ML-Algorithm-Recommendation-API-main/
│
├── app/
│   ├── main.py
│   │
│   ├── routes/
│   │   └── analyze.py
│   │
│   ├── services/
│   │   ├── dataset_analyzer.py
│   │   ├── algorithm_recommender.py
│   │   ├── ml_advisor.py
│   │   └── baseline_trainer.py
│   │
│   └── models/
│       └── response_models.py
│
├── frontend/
│   ├── public/
│   │
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   │
│   ├── package.json
│   └── vite.config.js
│
├── requirements.txt
└── README.md