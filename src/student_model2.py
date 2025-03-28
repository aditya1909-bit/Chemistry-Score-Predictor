import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import hamming_loss

# --- Data Loading ---
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student_data.csv")
df = pd.read_csv(data_path)

# Split the data:
#  - df_train: students who took the D test (DUCK TAKEN == 1)
#  - df_predict: students who haven't taken the D test (DUCK TAKEN == 0)
df_train = df[df["DUCK TAKEN"] == 1].copy()
df_predict = df[df["DUCK TAKEN"] == 0].copy()

# Clean column names: remove extra spaces and colons
df_train.columns = df_train.columns.str.strip().str.replace(":", "", regex=False)
df_predict.columns = df_predict.columns.str.strip().str.replace(":", "", regex=False)

# For training data, drop identifier columns not used as features.
df_train.drop(columns=["STUDENT ID", "DUCK TAKEN"], inplace=True)

# --- Define Column Groups ---
# D columns (e.g., D-1 to D-60) are our binary targets.
d_cols = [col for col in df_train.columns if col.startswith("D-")]

# E test columns (E0, E1, E2) will be used as features.
e_cols = [col for col in df_train.columns if col.startswith("E0-") or 
          col.startswith("E1-") or col.startswith("E2-")]

# Other numerical columns
num_cols = ["HS GPA", "SAT MATH", "SAT ENG", "SAT", "EMORY GPA", "CHEM GPA"]

# Categorical columns
cat_cols = ["UNBOUND", "ETHNICITY", "GENDER"]

# --- Prepare Features and Targets ---
X = df_train[cat_cols + num_cols + e_cols]
y = df_train[d_cols]  # All values are 0 or 1

# --- Preprocessing Pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), num_cols + e_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)

# --- Define Models and Parameter Grids ---
models = {
    "RandomForest": MultiOutputClassifier(RandomForestClassifier(random_state=42)),
    "XGBoost": MultiOutputClassifier(
        XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
    ),
    "NeuralNetwork": MultiOutputClassifier(
        MLPClassifier(max_iter=500, random_state=42)
    )
}

param_grids = {
    "RandomForest": {
         'classifier__estimator__n_estimators': [100, 200],
         'classifier__estimator__max_depth': [None, 10, 20]
    },
    "XGBoost": {
         'classifier__estimator__n_estimators': [100, 200],
         'classifier__estimator__max_depth': [3, 6, 10]
    },
    "NeuralNetwork": {
         'classifier__estimator__hidden_layer_sizes': [(128, 64), (256, 128)],
         'classifier__estimator__alpha': [0.0001, 0.001]
    }
}

# --- Split Training Data for Evaluation ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Function to Compute Per-Question Error Rates ---
def compute_error_rates(y_true, y_pred, columns):
    error_rates = {}
    # For each column (question), compute the misclassification rate.
    for idx, col in enumerate(columns):
        error_rate = (y_true.iloc[:, idx] != y_pred[:, idx]).mean()
        error_rates[col] = error_rate
    # Return as a sorted DataFrame (descending error rate)
    err_df = pd.DataFrame(list(error_rates.items()), columns=["Question", "Error Rate"])
    err_df.sort_values("Error Rate", ascending=False, inplace=True)
    return err_df

# --- Model Training, Hyperparameter Tuning, and Evaluation ---
for name, model in models.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    # Set up GridSearchCV for hyperparameter tuning.
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grids[name],
        cv=3,  # 3-fold cross-validation
        scoring="accuracy",  # scoring here is for overall accuracy (subset accuracy)
        verbose=0
    )
    
    # Fit grid search on training split.
    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_
    print(f"Best parameters for {name}: {grid.best_params_}")
    
    # Evaluate on validation set.
    subset_accuracy = best_pipeline.score(X_test, y_test)
    y_pred_val = best_pipeline.predict(X_test)
    ham_loss = hamming_loss(y_test, y_pred_val)
    per_label_accuracy = (y_test == y_pred_val).mean().mean()
    
    print(f"{name} subset accuracy on validation set: {subset_accuracy:.2f}")
    print(f"{name} Hamming loss on validation set: {ham_loss:.2f}")
    print(f"{name} average per-label accuracy: {per_label_accuracy:.2f}")
    
    # Compute error rates per D question.
    error_df = compute_error_rates(y_test, y_pred_val, d_cols)
    print(f"\n{name} - Top 5 questions with highest error rates:")
    print(error_df.head(5).to_string(index=False))
    
    # --- Prediction for Students Who Haven't Taken the D Test ---
    # Use the same preprocessor on df_predict.
    predict_features = df_predict[cat_cols + num_cols + e_cols]
    y_pred = best_pipeline.predict(predict_features)
    
    # Build predictions DataFrame with STUDENT ID.
    pred_df = pd.DataFrame(y_pred, columns=d_cols)
    pred_df.insert(0, "STUDENT ID", df_predict["STUDENT ID"].values)
    
    # Save predictions CSV into the data folder.
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", f"predictions_{name}.csv")
    pred_df.to_csv(output_path, index=False)
    print(f"Predictions saved for {name} in '{output_path}'")