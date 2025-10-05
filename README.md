📱 Telecom Customer Churn Prediction



🎯 Project Overview:-

Customer retention is one of the biggest challenges in the telecom industry. This project aims to predict customer churn — i.e., whether a customer will leave the service provider — based on historical data and service usage patterns.

By using machine learning models, we can identify customers who are likely to churn and help telecom companies take preventive actions such as offering promotions or improving service quality.

📊 Key Features: 

✅ Complete data preprocessing and cleaning
✅ Automatic detection of numeric & categorical columns
✅ Robust ML pipelines using Scikit-learn and Imbalanced-learn (SMOTE)
✅ Multiple models tested: Logistic Regression, Random Forest, and SMOTE-RF
✅ Hyperparameter tuning using GridSearchCV
✅ Model evaluation with ROC-AUC, PR-AUC, and Confusion Matrix
✅ Feature importance visualization
✅ Model persistence using Joblib
✅ Saves predictions for test data

🧠 Machine Learning Workflow:-

Data Import & Exploration
Uploaded dataset (Telco Customer Churn)
Inspected missing values, data types, and churn distribution
Data Cleaning
Converted TotalCharges to numeric
Handled missing values and duplicates
Removed customerID as it’s not a useful predictor
Feature Engineering
Automatic separation of numerical and categorical columns
Applied pipelines for imputation, scaling, and encoding
Model Building
Used Logistic Regression and Random Forest as base models
Implemented SMOTE to handle class imbalance
Hyperparameter Tuning
Applied GridSearchCV on the SMOTE + Random Forest pipeline
Evaluated using Stratified 5-Fold CV
Model Evaluation
Classification Report (Precision, Recall, F1-score)
ROC Curve, Precision-Recall Curve
Feature Importance visualization
Model Saving
Final model saved as churn_rf_model.joblib
Predictions exported as predictions_test.csv


🧩 Tech Stack
Category	Tools / Libraries
Programming Language	Python 3
Data Handling	pandas, numpy
Visualization	matplotlib, seaborn
ML & Pipelines	scikit-learn, imbalanced-learn
Model Tuning	GridSearchCV
Storage	joblib
Environment	Google Colab
📁 Folder Structure
Telecom-Churn-Prediction/
│
├── churn_rf_model.joblib       # Saved trained model
├── predictions_test.csv         # Model predictions on test data
├── telecom_churn.ipynb          # Main project notebook
├── README.md                    # Project documentation
└── data/                        # Dataset folder

📈 Evaluation Metrics: 
Metric	Description:-

ROC-AUC	Measures the model’s ability to distinguish churn vs non-churn
PR-AUC	Useful for imbalanced churn datasets
Confusion Matrix	Visualizes true vs predicted churn
Feature Importance	Highlights most influential factors
🔍 Example Output Visualizations

Churn Distribution

ROC Curve

Precision-Recall Curve

Top 25 Feature Importances

💾 How to Run
Option 1: Google Colab (Recommended)

Upload the notebook to Google Colab

Mount Google Drive:

from google.colab import drive
drive.mount('/content/drive')


Upload your dataset (.csv) when prompted

Run the cells in order

Option 2: Local Environment

Clone the repository:
git clone https://github.com/yourusername/telecom-churn-prediction.git

Install dependencies:
pip install -r requirements.txt


Run the script or notebook:
python telecom_churn.py

📊 Insights from the Model:

Customers with month-to-month contracts, electronic check payment, and no internet security are more likely to churn.
Long-term contracts and automatic payments are strong indicators of customer retention.

💡 Future Enhancements

🚀 Add XGBoost / LightGBM for better performance
📉 Deploy model using Streamlit or Flask
🧩 Build a churn prediction dashboard
📦 Integrate with real-time telecom data APIs
