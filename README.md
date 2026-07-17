# Predictive Modelling: Early Prediction of At-Risk Students (Python)

University of Otago, Master's Thesis (INFO580)

## Overview
This project builds and validates a machine learning pipeline to identify students at risk
of failing an introductory programming course (CS1), using mastery-learning activity data
from five cohort years. The goal is early, actionable prediction that instructors can use
to intervene before a student falls behind.

## What this demonstrates
- End-to-end ML pipeline design (data loading, feature engineering, modelling, evaluation)
- Leave-one-year-out cross-validation (LOYOCV) to prevent temporal data leakage
- Handling severe class imbalance with SMOTE
- Comparison across four classifiers: Logistic Regression, Random Forest, Gradient Boosting, and SVM
- Achieved 81% balanced accuracy across 1,154 records and 5 cohort years
- Permutation importance and statistical testing (Wilcoxon) to validate feature relevance and model stability
- Analysis across different temporal observation windows to test how early prediction remains reliable

**Business Impact:** Enables early identification of at-risk students, supporting timely interventions that can improve retention and success rates.

## Repository contents
- `Results/` – model outputs, LOYOCV results, and summary tables per classifier
- `Charts/` – visualizations of model performance, feature importance, and comparisons
- Key notebooks:
  1. `NB11_Outcome_Merge_and_Modelling.ipynb` – baseline pipeline and modelling setup
  2. `NB14_SMOTE_Analysis.ipynb` – class imbalance handling
  3. `NB21_Investigation_and_Visualisations.ipynb` – key results and window comparisons
  4. `NB23_Wilcoxon_Statistical_Tests.ipynb` – statistical validation of results

## Technologies
Python, scikit-learn, pandas, matplotlib, Jupyter Notebooks

## Data
The underlying student activity dataset is not included, as it contains data governed by
university ethics approval and cannot be shared publicly. Outputs, results, and code are
included so the methodology and findings can be reviewed in full.

## Notes
This was the technical foundation of my Master's thesis on early identification of at-risk
students using classification modelling.

## Pipeline Diagram

![Pipeline Diagram](pipeline_diagram.jpg) 
