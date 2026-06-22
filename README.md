🛡️ Credit Card Fraud Detection: Overcoming Extreme Class Imbalance

📖 Project Overview

In real-world financial data, fraudulent transactions are incredibly rare compared to legitimate purchases. This project builds a highly sensitive Machine Learning pipeline designed to detect credit card fraud in an extremely imbalanced dataset (where fraud accounts for only 0.172% of all transactions).

Instead of relying on standard accuracy metrics (which fail dangerously on imbalanced data), this project focuses on optimizing Recall and implementing specific mathematical techniques to force the AI to recognize minority class patterns.

🛠️ Core Engineering Pipeline

1. Data Engineering & Scaling

The Problem: Financial transactions have extreme outliers (e.g., a $50,000 corporate purchase vs. a $2 coffee). Standard MinMaxScaler algorithms get crushed by these outliers.

The Solution: Implemented RobustScaler, which utilizes the Interquartile Range (IQR) and Median to mathematically scale the Amount and Time features down to a normalized space without allowing massive transactions to break the 0-1 bounds.

2. Combating Imbalance with Geometry (SMOTE)

The Problem: The training matrix contained ~227,000 normal transactions and only ~394 fraud cases. An algorithm trained on this would default to predicting "Normal" 100% of the time.

The Solution: Utilized SMOTE (Synthetic Minority Over-sampling Technique). Instead of blindly duplicating data (which causes overfitting), SMOTE uses K-Nearest Neighbors (KNN) geometry to draw mathematical lines between existing fraud points and generate over 220,000 brand new, synthetic fraud rows. This perfectly leveled the training matrix to a 50/50 split.

3. The Brain: Extreme Gradient Boosting (XGBoost)

The Model: Deployed an XGBClassifier ensemble model.

The Engine: Trained using Logarithmic Loss (Cross-Entropy). This specific eval_metric mathematically destroys the AI's score if it is highly confident about a wrong answer, forcing it to output deeply calibrated probability percentages rather than just binary 1s and 0s.

4. Business Strategy: Threshold Tuning

By default, models trigger an alert at a 50% probability threshold. For fraud detection, waiting to be 50% sure means losing millions of dollars.

Extracted the raw predict_proba matrix and manually shifted the Decision Threshold down to 15%.

The Business Tradeoff: This shift slightly lowered Precision (causing more temporary card freezes for innocent customers) but massively spiked Recall (catching significantly more actual thieves), ultimately minimizing the total financial loss for the bank.

📁 Repository Structure

credit_card_fraud_project/
│
├── data/
│   ├── raw/                 # Downloaded kaggle dataset (ignored in git)
│
├── notebooks/
│   └── 01_fraud_detection.ipynb  # Core pipeline, SMOTE, and tuning
│
├── models/                  
│   └── xgboost_fraud_model.pkl   # Serialized model for deployment
│
├── requirements.txt         # Dependency list
└── README.md                # Project documentation

## Model Pipeline
1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Fraud Prediction
6. Performance Evaluation
