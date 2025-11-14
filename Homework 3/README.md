Homework 3 – Survival Models & CLV Analysis

This project applies parametric survival analysis and Customer Lifetime Value (CLV) modeling to a telecom churn dataset.
The goal is to understand churn behavior, identify factors that drive customer risk, and estimate CLV across different customer segments.

📂 Files in This Repository

Parametric_Models.py – fits AFT models (Weibull, Log-Logistic, Log-Normal, etc.), compares them, selects the final model, and performs feature selection.

CLV_Calculation.py – calculates CLV for each customer based on the final survival model and explores CLV across segments.

Report.md – main written report with interpretation, findings, and recommendations.

requirements.txt – Python packages required to run the project.

README.md – this file.

▶️ How to Run

Install dependencies:

pip install -r requirements.txt


Run the survival modeling file:

python Parametric_Models.py


Run the CLV calculation:

python CLV_Calculation.py

📝 Description

The analysis includes:

Fitting and comparing parametric AFT models

Identifying significant churn predictors

Calculating individual customer CLV

Investigating CLV differences across customer segments

Providing data-driven retention recommendations