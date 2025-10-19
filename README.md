# 📡 ConnectTel Customer Churn Prediction

Building a predictive model to identify customers likely to leave (churn) and providing actionable insights to enhance customer retention using data analytics and machine learning.

# Project Overview

This project focuses on developing a Customer Churn Prediction System for ConnectTel, a leading telecommunications provider offering broadband, mobile, and enterprise network solutions.
The company faced increasing customer attrition, leading to revenue loss and lower profitability.

As a data scientist, I analyzed customer behavior data, identified key churn drivers, and built predictive models capable of detecting at-risk customers with high accuracy.
The solution provides the business with early warnings and actionable retention strategies to reduce churn and improve customer satisfaction.

# Objectives

Predict which customers are likely to discontinue services.

Identify the key factors driving churn (e.g., contract type, payment method, tenure).

Evaluate multiple machine learning models to achieve optimal accuracy.

Recommend actionable retention strategies for management.

# Tools, Libraries & Tech Stack

Programming Language: Python

Libraries & Frameworks:

Pandas, NumPy — Data cleaning and preprocessing

Matplotlib, Seaborn — Data visualization

Scikit-learn — Machine learning model building and evaluation

XGBoost — Boosted ensemble learning

GridSearchCV, RandomizedSearchCV — Hyperparameter tuning

Jupyter Notebook — Interactive experimentation

Techniques:
EDA, Feature Engineering, Categorical Encoding, Scaling, Model Evaluation, Ensemble Learning

# Workflow / Steps to Execute
1️⃣ Data Collection

The dataset contained key customer information such as:
Gender, Tenure, PaymentMethod, InternetService, MonthlyCharges, Contract, OnlineSecurity, TechSupport, StreamingTV, Churn

2️⃣ Data Cleaning

Removed duplicates and handled missing values.

Normalized numerical variables using MinMaxScaler.

Encoded categorical features via One-Hot Encoding.

Split the dataset into training (80%) and test (20%) subsets.

3️⃣ Exploratory Data Analysis (EDA)

Visualized churn distribution (26.5% churn rate overall).

Found Monthly Contracts had the highest churn rate.

Discovered customers using Electronic Check were more likely to churn.

Identified that short-tenure customers exhibited a higher churn probability.

Explored correlations among numeric features (e.g., TotalCharges, Tenure, MonthlyCharges).

4️⃣ Feature Engineering

Created a new variable, Tenure Category, grouped as:

New Customers (0–12 months)

Mid-level Customers (13–36 months)

Long-term Customers (37+ months)

Improved model interpretability and performance using segmentation features.

5️⃣ Model Building

Trained and evaluated nine supervised learning algorithms, including:

Logistic Regression

Random Forest

Support Vector Classifier (SVC)

XGBoost

Decision Tree

Naive Bayes

K-Nearest Neighbors

SGD Classifier

6️⃣ Model Evaluation

Baseline Accuracy: Logistic Regression = 81.97%

After Hyperparameter Tuning (GridSearchCV): SGD Classifier = 82.19%

Final Ensemble (Voting Classifier): 81.26% Accuracy, improved Recall

7️⃣ Ensemble Optimization

Used Voting Classifier combining 7 models — Logistic Regression, Random Forest, XGBoost, SVC, Decision Tree, KNN, and Naive Bayes — with RandomizedSearchCV for optimal weighting.

# Results & Key Insights

Model	----------------------Accuracy	---------Precision	----------Recall	------------F1-Score	-----------ROC-AUC

Logistic Regression	--------81.97%	-----------68.89%	------------58.18%	------------63.08%	------------74.36%

Random Forest	------------80.98%	-----------67.44%	------------54.42%	------------60.24%	------------72.48%

SGD Classifier (Tuned)------82.19%	------------69.06%	---------59.25%	------------63.78%	------------74.85%

Voting Classifier (Final)----81.26%	------------81.33%	----------81.26%	------------81.30%	-------------76.11%

Key Observations:

Customers with short tenure and month-to-month contracts had higher churn risk.

Electronic check payment users were the most likely to leave.

Online security and tech support absence strongly correlated with churn.

Ensemble model achieved a strong trade-off between precision and recall.

# Business Recommendations

Retention Programs:
Offer loyalty discounts or contract upgrade incentives for month-to-month users.

Payment Channel Optimization:
Encourage churn-prone customers to switch from electronic checks to auto-pay or credit card billing.

Service Quality Improvement:
Enhance online security and tech support services to boost satisfaction.

Customer Segmentation:
Target marketing efforts at new and short-tenure customers for personalized onboarding.

Continuous Monitoring:
Implement churn dashboards to track KPIs such as churn rate, contract renewal rate, and customer satisfaction.

# Business Impact

Delivered a predictive system with 81% accuracy, reducing churn risk through early detection.

Enabled ConnectTel to retain an estimated 10–15% more customers via proactive intervention.

Provided actionable insights that informed marketing, customer success, and billing strategy decisions.

📁 Folder Structure

ConnectTel-Customer-Churn-Prediction/

│

├── data/

│   └── connecttel_churn.csv

│

├── notebooks/

│   ├── EDA.ipynb

│   ├── Model_Training.ipynb

│

├── visuals/

│   ├── churn_distribution.png

│   ├── correlation_matrix.png

│

├── models/

│   └── churn_model.pkl

│

├── README.md

└── requirements.txt

# Outcome Summary

This project successfully developed a machine learning–based churn prediction model that identifies at-risk telecom customers with high accuracy and supports strategic retention planning.
By combining EDA, feature engineering, and ensemble modeling, the project demonstrates expertise in data preprocessing, ML optimization, and business insight translation — core skills for any Data Scientist or ML Engineer.
