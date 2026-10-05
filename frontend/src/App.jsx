import { useState } from 'react'
import axios from 'axios'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  // =========================
  // FILE VALIDATION
  // =========================

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0]

    setError('')
    setResult(null)

    if (!selectedFile) {
      setFile(null)
      return
    }

    // Check file extension
    if (!selectedFile.name.toLowerCase().endsWith('.csv')) {
      setFile(null)
      setError(
        'Invalid file type. Please upload a CSV file only.'
      )
      return
    }

    // Check empty file
    if (selectedFile.size === 0) {
      setFile(null)
      setError(
        'The selected CSV file is empty. Please choose a file containing data.'
      )
      return
    }

    setFile(selectedFile)
  }

  // =========================
  // DATASET ANALYSIS
  // =========================

  const analyzeDataset = async () => {
    if (!file) {
      setError('Please select a CSV file first.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post(
        '/api/analyze-dataset',
        formData
      )

      setResult(response.data)
    } catch (err) {
      console.error('Dataset analysis error:', err)

      const detail = err.response?.data?.detail

      // FastAPI validation errors
      if (Array.isArray(detail)) {
        setError(
          detail
            .map(
              (item) =>
                item.msg || 'Validation error'
            )
            .join(', ')
        )
      }

      // FastAPI normal error message
      else if (typeof detail === 'string') {
        setError(detail)
      }

      // Backend not running / network problem
      else if (
        err.code === 'ERR_NETWORK' ||
        !err.response
      ) {
        setError(
          'Unable to connect to the backend. Please make sure the FastAPI server is running.'
        )
      }

      // Other errors
      else {
        setError(
          'Something went wrong while analyzing the dataset. Please try again.'
        )
      }
    } finally {
      setLoading(false)
    }
  }

  // =========================
  // FORMAT METRICS
  // =========================

  const formatMetric = (value) => {
    if (value === null || value === undefined) {
      return 'N/A'
    }

    if (typeof value === 'number') {
      return value.toFixed(4)
    }

    return value
  }

  // =========================
  // RESULT DATA
  // =========================

  const summary = result?.dataset_summary
  const comparisons = result?.model_comparison || []

  const isClassification =
    summary?.problem_type === 'classification'

  /*
    Classification:
    F1 Score is used for the comparison chart.

    Regression:
    RMSE is used for the comparison chart.
  */

  const chartMetric = isClassification
    ? 'f1_score'
    : 'rmse'

  const chartMetricLabel = isClassification
    ? 'F1 Score'
    : 'RMSE'

  const chartData = comparisons
    .filter(
      (item) =>
        item[chartMetric] !== null &&
        item[chartMetric] !== undefined
    )
    .map((item) => ({
      model: item.model,
      value: item[chartMetric],
    }))

  return (
    <div className="app">

      {/* =========================
          HERO SECTION
      ========================= */}

      <header className="hero-section">
        <div className="hero-content">

          <span className="badge">
            MACHINE LEARNING TOOL
          </span>

          <h1>
            ML Algorithm Recommender
          </h1>

          <p>
            Upload a CSV dataset and automatically
            analyze the problem type, recommend ML
            algorithms, compare models, and identify
            the best performing model.
          </p>

        </div>
      </header>

      <main className="container">

        {/* =========================
            UPLOAD SECTION
        ========================= */}

        <section className="upload-card">

          <div className="section-title">
            <h2>
              Analyze Your Dataset
            </h2>

            <p>
              Select a CSV file to start the analysis.
            </p>
          </div>

          <div className="upload-area">

            <input
              id="file-upload"
              type="file"
              accept=".csv"
              onChange={handleFileChange}
            />

            <label
              htmlFor="file-upload"
              className="file-label"
            >

              <span className="upload-icon">
                ↑
              </span>

              <strong>
                {file
                  ? file.name
                  : 'Choose a CSV file'}
              </strong>

              <small>
                {file
                  ? `${(
                      file.size / 1024
                    ).toFixed(1)} KB selected`
                  : 'CSV files only'}
              </small>

            </label>
          </div>

          {/* ANALYZE BUTTON */}

          <button
            className="analyze-button"
            onClick={analyzeDataset}
            disabled={!file || loading}
          >
            {loading ? (
              <span className="loading-content">
                <span className="spinner"></span>
                Analyzing Dataset...
              </span>
            ) : (
              'Analyze Dataset'
            )}
          </button>

          {/* ERROR MESSAGE */}

          {error && (
            <div
              className="error-message"
              role="alert"
            >
              <strong>
                Analysis Error
              </strong>

              <p>
                {error}
              </p>
            </div>
          )}

        </section>

        {/* =========================
            RESULTS
        ========================= */}

        {result && (
          <>

            {/* =========================
                RESULTS HEADER
            ========================= */}

            <section className="results-header">

              <div>

                <span className="badge">
                  ANALYSIS COMPLETE
                </span>

                <h2>
                  Dataset Analysis Results
                </h2>

              </div>

              <div className="problem-badge">
                {summary?.problem_type?.toUpperCase()}
              </div>

            </section>

            {/* =========================
                DATASET SUMMARY
            ========================= */}

            <section className="summary-grid">

              <div className="info-card">
                <span>Rows</span>

                <strong>
                  {summary?.rows ?? 'N/A'}
                </strong>
              </div>

              <div className="info-card">
                <span>Columns</span>

                <strong>
                  {summary?.columns ?? 'N/A'}
                </strong>
              </div>

              <div className="info-card">
                <span>Problem Type</span>

                <strong>
                  {summary?.problem_type ?? 'N/A'}
                </strong>
              </div>

              <div className="info-card">
                <span>Target Column</span>

                <strong>
                  {summary?.target_column ?? 'N/A'}
                </strong>
              </div>

            </section>

            {/* =========================
                RECOMMENDED ALGORITHMS
            ========================= */}

            <section className="content-card">

              <div className="section-title">

                <h2>
                  Recommended Algorithms
                </h2>

                <p>
                  Algorithms suggested for this dataset.
                </p>

              </div>

              <div className="algorithm-grid">

                {result.recommended_algorithms?.map(
                  (item, index) => (

                    <div
                      className="algorithm-card"
                      key={index}
                    >

                      <div className="algorithm-number">
                        {index + 1}
                      </div>

                      <div>

                        <h3>
                          {item.algorithm}
                        </h3>

                        <p>
                          {item.reason}
                        </p>

                      </div>

                    </div>
                  )
                )}

              </div>

            </section>

            {/* =========================
                SUGGESTED METRICS
            ========================= */}

            <section className="content-card">

              <div className="section-title">

                <h2>
                  Suggested Metrics
                </h2>

              </div>

              <div className="metric-tags">

                {result.suggested_metrics?.map(
                  (metric, index) => (

                    <span key={index}>
                      {metric}
                    </span>

                  )
                )}

              </div>

            </section>

            {/* =========================
                PREPROCESSING
            ========================= */}

            <section className="content-card">

              <div className="section-title">

                <h2>
                  Preprocessing Steps
                </h2>

              </div>

              <div className="preprocessing-list">

                {result.preprocessing_steps?.map(
                  (step, index) => (

                    <div
                      className="preprocessing-item"
                      key={index}
                    >

                      <span>
                        ✓
                      </span>

                      <p>
                        {step}
                      </p>

                    </div>

                  )
                )}

              </div>

            </section>

            {/* =========================
                CLASS IMBALANCE
            ========================= */}

            {result.class_imbalance && (
              <section className="content-card">

                <div className="section-title">

                  <h2>
                    Class Imbalance
                  </h2>

                </div>

                <div className="imbalance-box">

                  <div>

                    <span>
                      Status
                    </span>

                    <strong>
                      {result.class_imbalance
                        .is_imbalanced
                        ? 'Imbalanced'
                        : 'Balanced'}
                    </strong>

                  </div>

                  <div className="distribution">

                    <span>
                      Class Distribution
                    </span>

                    {Object.entries(
                      result.class_imbalance
                        .class_distribution || {}
                    ).map(
                      ([className, percentage]) => (

                        <div
                          className="distribution-row"
                          key={className}
                        >

                          <span>
                            Class {className}
                          </span>

                          <strong>
                            {(
                              percentage * 100
                            ).toFixed(1)}
                            %
                          </strong>

                        </div>

                      )
                    )}

                  </div>

                </div>

              </section>
            )}

            {/* =========================
                MODEL COMPARISON
            ========================= */}

            <section className="content-card">

              <div className="section-title">

                <h2>
                  Model Comparison
                </h2>

                <p>
                  Performance of the evaluated models.
                </p>

              </div>

              {/* =========================
                  COMPARISON CHART
              ========================= */}

              {chartData.length > 0 && (

                <div className="chart-container">

                  <div className="chart-title">

                    <h3>
                      {chartMetricLabel} Comparison
                    </h3>

                    <p>
                      {isClassification
                        ? 'Higher F1 Score indicates better performance.'
                        : 'Lower RMSE indicates better performance.'}
                    </p>

                  </div>

                  <ResponsiveContainer
                    width="100%"
                    height={340}
                  >

                    <BarChart
                      data={chartData}
                      margin={{
                        top: 20,
                        right: 20,
                        left: 10,
                        bottom: 75,
                      }}
                    >

                      <CartesianGrid
                        strokeDasharray="3 3"
                      />

                      <XAxis
                        dataKey="model"
                        angle={-15}
                        textAnchor="end"
                        interval={0}
                        height={90}
                      />

                      <YAxis />

                      <Tooltip
                        formatter={(value) => [
                          Number(value).toFixed(4),
                          chartMetricLabel,
                        ]}
                      />

                      <Bar
                        dataKey="value"
                        name={chartMetricLabel}
                        radius={[
                          6,
                          6,
                          0,
                          0,
                        ]}
                      />

                    </BarChart>

                  </ResponsiveContainer>

                </div>
              )}

              {/* =========================
                  COMPARISON TABLE
              ========================= */}

              {comparisons.length > 0 ? (

                <div className="table-wrapper">

                  <table>

                    <thead>

                      <tr>

                        <th>
                          Model
                        </th>

                        {isClassification ? (
                          <>
                            <th>
                              Accuracy
                            </th>

                            <th>
                              F1 Score
                            </th>
                          </>
                        ) : (
                          <>
                            <th>
                              RMSE
                            </th>

                            <th>
                              MAE
                            </th>

                            <th>
                              R² Score
                            </th>
                          </>
                        )}

                      </tr>

                    </thead>

                    <tbody>

                      {comparisons.map(
                        (model, index) => (

                          <tr key={index}>

                            <td className="model-name">
                              {model.model}
                            </td>

                            {isClassification ? (
                              <>
                                <td>
                                  {formatMetric(
                                    model.accuracy
                                  )}
                                </td>

                                <td>
                                  {formatMetric(
                                    model.f1_score
                                  )}
                                </td>
                              </>
                            ) : (
                              <>
                                <td>
                                  {formatMetric(
                                    model.rmse
                                  )}
                                </td>

                                <td>
                                  {formatMetric(
                                    model.mae
                                  )}
                                </td>

                                <td>
                                  {formatMetric(
                                    model.r2_score
                                  )}
                                </td>
                              </>
                            )}

                          </tr>

                        )
                      )}

                    </tbody>

                  </table>

                </div>

              ) : (

                <p>
                  No model comparison results available.
                </p>

              )}

            </section>

            {/* =========================
                BEST MODEL
            ========================= */}

            {result.best_model && (

              <section className="best-model-card">

                <span className="badge">
                  BEST MODEL
                </span>

                <h2>
                  {result.best_model}
                </h2>

                <p>
                  Selected using{' '}
                  <strong>
                    {result.selection_metric}
                  </strong>
                  .
                </p>

              </section>

            )}

          </>
        )}

      </main>

      {/* =========================
          FOOTER
      ========================= */}

      <footer>

        <p>
          ML Algorithm Recommender API
        </p>

        <span>
          React + FastAPI + Scikit-learn
        </span>

      </footer>

    </div>
  )
}

export default App