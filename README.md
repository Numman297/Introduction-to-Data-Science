# Student Sleep Habits and Mental Health: Impact on Academic Performance

[![R Version](https://img.shields.io/badge/R-%E2%89%A5%204.2.0-blue.svg?logo=r)](https://www.r-project.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](#)
[![Dataset](https://img.shields.io/badge/Dataset-3000%20Records-orange.svg)](#dataset-overview)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Data Science and Machine Learning research investigation exploring the multidimensional relationships between sleep hygiene, recreational screen time, psychological distress, and cumulative Grade Point Average (GPA) among higher education students.

---

## 👥 Project Authors & Contributors

| Name | Student ID | GitHub Profile |
| :--- | :--- | :--- |
| **MD. NUMMAN** | `23-54538-3` | [@Numman297](https://github.com/Numman297) |
| **S.T.M. MUSTAFA TAUSIF** | `23-51469-1` | — |
| **MD. WASIF HASAN** | `23-50938-1` | — |
| **MD. SABBIR HOSSAIN ROHAN** | `23-50753-1` | — |

---

## 📌 Table of Contents

- [Executive Summary](#-executive-summary)
- [Research Questions](#-research-questions)
- [Dataset Overview](#-dataset-overview)
- [Data Science Pipeline & Methodology](#-data-science-pipeline--methodology)
  - [1. Data Acquisition & Exploration (EDA)](#1-data-acquisition--exploration-eda)
  - [2. Preprocessing & Cleaning](#2-preprocessing--cleaning)
  - [3. Feature Engineering & Selection](#3-feature-engineering--selection)
  - [4. Dimensionality Reduction (PCA)](#4-dimensionality-reduction-pca)
  - [5. Machine Learning Models](#5-machine-learning-models)
- [Model Evaluation & Benchmark](#-model-evaluation--benchmark)
- [Key Insights & Domain Takeaways](#-key-insights--domain-takeaways)
- [Repository Structure](#-repository-structure)
- [Limitations & Ethical Considerations](#-limitations--ethical-considerations)
- [License](#-license)

---

## 📖 Executive Summary

Academic performance in higher education is fundamentally constrained by physiological rest and psychological equilibrium. Leveraging a curated dataset of **3,000 student records across 15 multidimensional variables**, this project builds a robust analytical and predictive pipeline to determine how daily lifestyle factors affect academic outcomes.

Our empirical findings reveal:
- **Sleep duration and psychological strain** account for **>44% of cumulative GPA variance** ($\text{Test } R^2 \approx 0.448$, $\text{RMSE} \approx 0.2917$).
- Average sleep duration retains a **statistically significant positive association with GPA** ($p < 0.001$), even after rigorously controlling for study hours and digital screen exposure (**RQ1**).
- Recreational screen time exerts a **dual negative pathway**: it directly elevates stress and anxiety scores while displacing restorative sleep hours (**RQ3**).
- Severe psychological distress ($\text{Mental Load Index} \ge 8$) imposes an academic penalty of up to **$-0.50$ GPA grade points** (**RQ2**).

---

## 🎯 Research Questions

| ID | Research Question | Validation Outcome |
| :---: | :--- | :---: |
| **RQ1** | Does average sleep duration have a significant positive association with GPA, even after controlling for study hours and recreational screen time? | ✅ **Validated** ($\beta > 0, p < 0.001$ across all screen & study tiers) |
| **RQ2** | Do students with elevated stress and anxiety scores show measurably lower GPA and higher incidence of burnout? | ✅ **Validated** (Significant negative gradient, $-0.35$ to $-0.50$ grade penalty) |
| **RQ3** | Is there an inverse relationship between screen time / social media engagement and sleep duration, indicating digital displacement? | ✅ **Validated** (Strong negative correlation $r = -0.59$) |

---

## 📊 Dataset Overview

- **Source File:** [`student_sleep_mental_health_2026.csv`](student_sleep_mental_health_2026.csv)
- **Observations:** 3,000 students
- **Features:** 15 variables (Continuous, Categorical, Boolean)

### Feature Breakdown

| Category | Variable | Type | Description |
| :--- | :--- | :--- | :--- |
| **Identifier** | `student_id` | String | Unique student identifier |
| **Demographics** | `age` | Integer | Student age (years) |
| | `gender` | Categorical | Male, Female, Non-binary, Prefer not to say |
| | `education_level` | Categorical | High School, Undergraduate, Postgraduate |
| **Lifestyle & Habits** | `avg_sleep_hours` | Numeric | Average daily sleep duration (hours) |
| | `screen_time_hours` | Numeric | Total daily recreational & device screen time (hours) |
| | `social_media_hours` | Numeric | Daily time spent on social media platforms (hours) |
| | `study_hours_per_day` | Numeric | Dedicated daily study duration (hours) |
| | `exercise_hours_per_week` | Numeric | Weekly physical activity duration (hours) |
| | `caffeine_drinks_per_day` | Numeric | Daily caffeinated beverages consumed |
| **Psychological Metrics** | `stress_level` | Integer | Standardized self-reported stress (1–10) |
| | `anxiety_score` | Integer | Standardized clinical anxiety screening score (1–10) |
| | `feels_burned_out` | Boolean | Academic burnout self-assessment (TRUE / FALSE) |
| | `uses_sleep_app` | Boolean | Usage of sleep-tracking / meditation applications |
| **Target Variable** | `gpa` | Continuous | Cumulative Grade Point Average ($2.00 - 4.00$ scale) |

---

## ⚙️ Data Science Pipeline & Methodology

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Raw Data & CSV │ ──> │   EDA & Skewness     │ ──> │ Data Preprocessing   │
│  (3,000 rows)   │     │ Inspection & Heatmap │     │ (Impute, Cap, Scale) │
└─────────────────┘     └──────────────────────┘     └──────────────────────┘
                                                                 │
                                                                 ▼
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│ Model Benchmark │ <── │  Supervised ML       │ <── │ Feature Engineering  │
│ & Evaluation    │     │  (OLS, CART, RF, PCA)│     │ & PCA Reduction      │
└─────────────────┘     └──────────────────────┘     └──────────────────────┘
```

### 1. Data Acquisition & Exploration (EDA)
- **Univariate Analysis:** Evaluated distribution shapes, histograms, and skewness coefficients across all continuous parameters using the `moments` package. Identified moderate positive skewness in `caffeine_drinks_per_day` ($0.782$).
- **Bivariate & Multicollinearity Analysis:** Computed full correlation matrix with `ggcorrplot`. Identified high collinearity between `stress_level` and `anxiety_score` ($r = 0.74$) as well as `screen_time_hours` and `social_media_hours` ($r = 0.77$).

### 2. Preprocessing & Cleaning
- **Missing Value Imputation:** Handled Missing Completely at Random (MCAR) values in `social_media_hours` (14), `exercise_hours_per_week` (14), and `anxiety_score` (7) using robust **median imputation**.
- **Outlier Treatment:** Applied **IQR-based Winsorization** to cap extreme values in `avg_sleep_hours`, `screen_time_hours`, and `study_hours_per_day` without sacrificing valid data signals.
- **Encoding & Normalization:** Categorical predictors (`gender`, `education_level`) were one-hot encoded via `model.matrix()`; continuous predictors were standardized ($Z$-score scaling) using `scale()`.

### 3. Feature Engineering & Selection
- **`screen_to_sleep_ratio`:** Quantifies digital displacement relative to biological sleep duration.
- **`mental_load_index`:** Composite index $\frac{\text{stress\_level} + \text{anxiety\_score}}{2}$ resolving collinearity into a unified distress metric.
- **`age_group`:** Binned age cohorts (Teen, Young Adult, Adult) capturing potential non-linear life-stage patterns.
- **Feature Selection:** Dropped low-correlation variables (`age`, `caffeine_drinks_per_day`) while explicitly retaining `study_hours_per_day` as a theoretical control.

### 4. Dimensionality Reduction (PCA)
- Principal Component Analysis on continuous features demonstrated that **PC1** ("Digital Overuse vs. Sleep Depletion Axis") and **PC2** ("Affective Strain vs. Physical Activity Axis") explain **67.91%** of cumulative variance.
- Retaining 4 orthogonal components under the Kaiser Criterion captured **88.33%** of total multivariate variance.

### 5. Machine Learning Models
Models were trained using an **80% training ($n=2,401$) / 20% hold-out test ($n=599$)** split with **10-fold cross-validation**:
1. **Baseline Multiple Linear Regression (OLS):** Highly interpretable benchmark model.
2. **CART Decision Tree (`rpart`):** Tuned across complexity parameters ($\text{cp}$) with pruning.
3. **Random Forest Ensemble (`randomForest`):** Ensembled 200 regression trees tuning candidate split parameters ($\text{mtry}$).
4. **PCA-Transformed Regression:** Regression on top 4 orthogonal principal components to assess multicollinearity remedies.

---

## 📈 Model Evaluation & Benchmark

All models were evaluated on the independent hold-out test set ($n = 599$) against our primary criteria: $\text{RMSE} < 0.35$ and $R^2 \ge 0.40$.

| Model | CV RMSE | CV $R^2$ | Test RMSE | Test MAE | Test $R^2$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Linear Regression** | **0.2872** | **0.4431** | **0.2917** | **0.2330** | **0.4485** | 🏆 **Top Performer** |
| **Random Forest Ensemble** | 0.2923 | 0.4285 | 0.2979 | 0.2384 | 0.4247 | Strong Non-linear Fit |
| **PCA Linear Regression (4 PCs)** | 0.2898 | 0.4350 | 0.2940 | 0.2356 | 0.4397 | Robust Orthogonal Fit |
| **Decision Tree (CART)** | 0.3120 | 0.3512 | 0.3160 | 0.2523 | 0.3524 | Parsimonious Rules |

> **Key Takeaway:** The Multiple Linear Regression model with engineered composite metrics achieved the lowest Test RMSE ($0.2917$) and highest explanatory power ($R^2 = 44.85\%$), indicating that the relationship between sleep, distress, and GPA is predominantly continuous and linear once non-linear ratios are explicitly framed.

---

## 💡 Key Insights & Domain Takeaways

1. **Sleep Is an Academic Multiplier:** Each additional hour of nightly sleep (up to 8.5 hours) correlates with a $+0.08$ to $+0.12$ gain in cumulative GPA.
2. **Study Hours Cannot Offset Chronic Sleep Loss:** The positive return on sleep holds consistently true across both high-study and low-study student cohorts.
3. **The Screen Time Multi-Trap:** Recreational digital screen time greater than 8.5 hours/day correlates with reduced sleep duration and compounded psychological anxiety.
4. **Actionable Policy Interventions:** Universities should prioritize institutional sleep hygiene education, digital mindfulness workshops, and early burnout screening rather than solely pushing study hours.

---

## 📁 Repository Structure

```
Introduction-to-Data-Science/
├── J05Final.Rmd                         # Complete R Markdown research manuscript & code
├── J05Final.html                        # Knitted, interactive HTML report with charts & tables
├── student_sleep_mental_health_2026.csv  # Primary dataset (3,000 student survey records)
├── README.md                            # Comprehensive project documentation
```


---

## ⚠️ Limitations & Ethical Considerations

- **Observational Nature:** The survey data is cross-sectional; causality cannot be definitively proven without randomized trials or longitudinal sensor tracking.
- **Self-Reporting Bias:** Metrics such as study hours and stress levels are subject to recall and social desirability biases.
- **Ethical Use:** Predictive models of GPA must never be used punitively for admissions or scholarship revocations. Wellness data must remain anonymous and decoupled from academic sanction.

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use, modify, and reference this work with proper attribution.
