# Python 3.9+ example (scikit-learn, xgboost)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (roc_auc_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, brier_score_loss,
                             precision_recall_curve, auc)
import shap
import matplotlib.pyplot as plt
import joblib
# 1. Load data
df = pd.read_csv("hospital_discharges.csv", parse_dates=['admission_date','discharge_date'])
# 2. Basic cleaning
df = df.drop_duplicates(subset=['index_admission_id'])
# Define target
df['readmit_30d'] = df['readmit_30d'].astype(int)
# 3. Feature list (example)
numeric_features = ['age', 'los_index', 'charlson_index', 'num_prior_admissions_6mo', 'num_medications_at_discharge']
categorical_features = ['sex', 'insurance_type', 'discharge_disposition']  # add as needed
binary_features = ['abnormal_labs_flag', 'socioeconomic_flag']
# 4. Preprocessing pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
])
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_features),
    ('cat', cat_pipeline, categorical_features)
], remainder='passthrough')  # passthrough binary features
# 5. Train-test split
X = df[numeric_features + categorical_features + binary_features]
y = df['readmit_30d']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42)
# 6. Model pipelines
lr_pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])
rf_pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                   random_state=42, n_jobs=-1))
])
xgb_pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              random_state=42, n_jobs=-1))
])
# 7. Quick baseline training
lr_pipe.fit(X_train, y_train)
rf_pipe.fit(X_train, y_train)
xgb_pipe.fit(X_train, y_train)
# 8. Predictions and metrics function
def evaluate_model(pipe, X_test, y_test, name="Model"):
    y_proba = pipe.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    brier = brier_score_loss(y_test, y_proba)
    print(f"{name} AUC: {auc:.3f}  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}  Brier: {brier:.3f}")
    print(classification_report(y_test, y_pred, digits=3))
    cm = confusion_matrix(y_test, y_pred)
    return {'auc':auc, 'precision':precision, 'recall':recall, 'f1':f1, 'brier':brier, 'cm':cm, 'y_proba':y_proba}
res_lr = evaluate_model(lr_pipe, X_test, y_test, "Logistic Regression")
res_rf = evaluate_model(rf_pipe, X_test, y_test, "Random Forest")
res_xgb = evaluate_model(xgb_pipe, X_test, y_test, "XGBoost")
# 9. Feature importance for XGBoost (requires extracting preprocessed feature names)
# NOTE: Produce feature names from transformer
ohe = xgb_pipe.named_steps['pre'].transformers_[1][1].named_steps['ohe']
ohe_cols = ohe.get_feature_names_out(categorical_features)
feature_names = numeric_features + list(ohe_cols) + binary_features
booster = xgb_pipe.named_steps['clf'].get_booster()
imp = booster.get_score(importance_type='gain')
# Map and sort
imp_df = pd.DataFrame([(k, imp.get(k,0)) for k in imp.keys()], columns=['feature','gain']).sort_values('gain', ascending=False)
print(imp_df.head(20))
# 10. SHAP (on a sample)
explainer = shap.Explainer(xgb_pipe.named_steps['clf'])
X_sample = preprocessor.transform(X_test.sample(500, random_state=1))
shap_values = explainer(X_sample)
shap.summary_plot(shap_values, features=X_sample, feature_names=feature_names)
# 11. Save best model
joblib.dump(xgb_pipe, "xgb_readmit_pipeline.joblib")
