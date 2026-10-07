import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ==========================================
# 1. LOAD DATASET
# ==========================================
# Download 'creditcard.csv' from Kaggle and keep it in your working directory
df = pd.read_csv('creditcard.csv')

print("Dataset Shape:", df.shape)
print("\nClass Distribution:\n", df['Class'].value_counts())

# ==========================================
# 2. DATA PREPROCESSING & FEATURE SCALING
# ==========================================
# Check for missing valuesś
print("\nMissing values:", df.isnull().sum().max())

# Scale 'Time' and 'Amount' features as V1-V28 are already PCA transformed
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop original unscaled columns
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Separate features (X) and target variable (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Stratified Train-Test Split to maintain class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# ==========================================
# 3. HANDLING IMBALANCED DATA (SMOTE)
# ==========================================
# Apply Synthetic Minority Over-sampling Technique ONLY on training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTE (Train set):")
print(pd.Series(y_train_res).value_counts())

# ==========================================
# 4. CLASSIFICATION MODELS
# ==========================================

# Helper function for evaluation metrics
def evaluate_model(model, name, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n================ {name} ================")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # ROC-AUC & AUPRC Score
    roc_auc = roc_auc_score(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    auprc = auc(recall, precision)
    
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC (AUPRC) Score: {auprc:.4f}")
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Fraud', 'Fraud'], 
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

# --- Model 1: Logistic Regression ---
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_res, y_train_res)
evaluate_model(lr_model, "Logistic Regression (with SMOTE)", X_test, y_test)

# --- Model 2: Random Forest Classifier ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_res, y_train_res)
evaluate_model(rf_model, "Random Forest Classifier (with SMOTE)", X_test, y_test)