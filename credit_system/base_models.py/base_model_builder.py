from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def build_standard_models(categorical_cols, numerical_cols):
    """Build standard sklearn preprocessing and model pipelines"""
    # Create a column transformer for numeric and categorical features
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])
    
    # Logistic Regression Pipeline
    logreg_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced"))
    ])
    
    # Calibrated Logistic Regression (better probability estimates)
    calibrated_logreg_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", CalibratedClassifierCV(
            LogisticRegression(max_iter=5000, class_weight="balanced"),
            cv=5, method="isotonic"))
    ])
    
    # XGBoost Pipeline
    xgb_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", XGBClassifier(n_jobs=1, use_label_encoder=False, eval_metric="logloss", 
                                    scale_pos_weight=1, random_state=42))
    ])
    
    # Random Forest Pipeline
    rf_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42, max_depth=10))
    ])
    
    return {
        "logistic_regression": logreg_pipeline,
        "calibrated_logistic_regression": calibrated_logreg_pipeline,
        "xgboost": xgb_pipeline,
        "random_forest": rf_pipeline
    }, preprocessor