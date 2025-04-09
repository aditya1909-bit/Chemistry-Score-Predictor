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
from sklearn.metrics import hamming_loss, make_scorer

print("\nRunning student_model3.py...")

# --- Data Loading ---
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "updated_student_data.csv")
df = pd.read_csv(data_path)

# --- Define All Column Groups Based on Provided Header ---
# Identifier columns:
id_cols = ["STUDENT ID"]
# DUCK TAKEN is used for splitting
duck_col = "DUCK TAKEN"
# UNBOUND (we keep as categorical)
unbound_col = "UNBOUND"

# Columns between HS GPA and GENDER (these are part of features and a 0 means no data)
num_cols = ["HS GPA", "SAT MATH", "SAT ENG", "SAT", "EMORY GPA", "CHEM GPA"]
cat_cols_basic = ["ETHNICITY", "GENDER"]

# Define the grade columns (class grades from 100 up to 729R)
grade_cols = [
    "100", "120", "141", "142", "150", "157", "190", "202", "203", "204", "205",
    "221", "222", "260", "261", "300", "301", "322", "327", "328", "331", "332", "333",
    "340", "343", "350", "370", "470", "521", "522", "523", "524", "531", "532", "533",
    "534", "536", "551", "571", "572", "574", "142Q", "150L", "202L", "202Q", "202Z",
    "202ZL", "203L", "203Q", "203Z", "203ZL", "204L", "204LE", "205L", "205LE", "221L",
    "221Z", "222L", "222Z", "226L", "227L", "260L", "260W", "300L", "327L", "331LW",
    "332LW", "335LW", "355L", "370W", "371L", "371LW", "392R", "397R", "399R", "399RE",
    "468W", "475R", "495A", "495BW", "495RW", "496R", "497R", "499R", "575R", "729R"
]

# E test columns (E0-, E1-, E2-)
e_cols = [col for col in df.columns if col.startswith("E0-") or 
          col.startswith("E1-") or col.startswith("E2-")]

# --- Clean Column Names ---
# Remove extra spaces and colons from the entire dataframe’s columns.
df.columns = df.columns.str.strip().str.replace(":", "", regex=False)

# --- Split the Data ---
# Training data: students who took the D test (DUCK TAKEN == 1)
# Prediction data: students who have not taken the D test (DUCK TAKEN == 0)
df_train = df[df[duck_col] == 1].copy()
df_predict = df[df[duck_col] == 0].copy()

# --- Replace 0 with Missing Values (np.nan) for Specific Columns ---
# The following columns treat a 0 as "no data": numeric columns from HS GPA to CHEM GPA,
# categorical columns ETHNICITY and GENDER, and all grade columns.
cols_where_zero_is_missing = num_cols + cat_cols_basic + grade_cols

for d in [df_train, df_predict]:
    # Replace numeric 0 with np.nan. For non-numeric columns (e.g., grades, ethnicity, gender),
    # the replacement will affect "0" string entries as well.
    d[cols_where_zero_is_missing] = d[cols_where_zero_is_missing].replace(0, np.nan)

# For training data, drop identifier columns and the DUCK TAKEN flag.
df_train.drop(columns=[id_cols[0], duck_col], inplace=True)

# --- Define Target and Features ---
# Targets: All columns starting with "D-" (D test questions)
d_cols = [col for col in df_train.columns if col.startswith("D-")]

# Features: Combine categorical and numerical features.
# We update categorical features to include unbound, ethnicity, gender, and all grade columns.
cat_cols = [unbound_col] + cat_cols_basic + grade_cols

# X will include categorical features, numerical features, and E columns.
X = df_train[cat_cols + num_cols + e_cols]
y = df_train[d_cols]  # All D columns are binary targets

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

# --- Define a Custom Scorer for Average Per-Label Accuracy ---
def per_label_accuracy_score(y_true, y_pred):
    # Compute the average per-label accuracy across all target columns
    return (y_true == y_pred).mean().mean()

scorer = make_scorer(per_label_accuracy_score, greater_is_better=True)

# --- Define Models and Parameter Grids ---
models = {
    "RandomForest": MultiOutputClassifier(RandomForestClassifier(random_state=42)),
    "XGBoost": MultiOutputClassifier(
        XGBClassifier(random_state=42, eval_metric="logloss")
    ),
    "NeuralNetwork": MultiOutputClassifier(
        MLPClassifier(random_state=42)
    )
}

param_grids = {
    "RandomForest": {
         'classifier__estimator__n_estimators': [200, 300],
         'classifier__estimator__max_depth': [None, 20, 30]
    },
    "XGBoost": {
         'classifier__estimator__n_estimators': [200, 300],
         'classifier__estimator__max_depth': [3, 6, 10, 15]
    },
    "NeuralNetwork": {
         'classifier__estimator__hidden_layer_sizes': [(128, 64), (256, 128), (256, 128, 64)],
         'classifier__estimator__alpha': [0.0001, 0.001],
         'classifier__estimator__max_iter': [500, 800]
    }
}

# --- Split Training Data for Evaluation ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Function to Compute Per-Question Error Rates ---
def compute_error_rates(y_true, y_pred, columns):
    error_rates = {}
    # For each target column, compute the misclassification rate.
    for idx, col in enumerate(columns):
        error_rate = (y_true.iloc[:, idx] != y_pred[:, idx]).mean()
        error_rates[col] = error_rate
    # Return as a sorted DataFrame (highest error rate at the top)
    err_df = pd.DataFrame(list(error_rates.items()), columns=["Question", "Error Rate"])
    err_df.sort_values("Error Rate", ascending=False, inplace=True)
    return err_df

# --- Initialize Best Model Variables ---
best_accuracy = -1
best_name = None
best_pipeline_final = None

# --- Model Training, Hyperparameter Tuning, and Evaluation ---
for name, model in models.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    # Set up GridSearchCV for hyperparameter tuning using the custom scorer.
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grids[name],
        cv=3,  # 3-fold cross-validation
        scoring=scorer,
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
    per_label_accuracy = per_label_accuracy_score(y_test, y_pred_val)
    
    print(f"{name} subset accuracy on validation set: {subset_accuracy:.2f}")
    print(f"{name} Hamming loss on validation set: {ham_loss:.2f}")
    print(f"{name} average per-label accuracy: {per_label_accuracy:.2f}")
    
    if per_label_accuracy > best_accuracy:
        best_accuracy = per_label_accuracy
        best_name = name
        best_pipeline_final = best_pipeline
    
    # Compute error rates per D question.
    error_df = compute_error_rates(y_test, y_pred_val, d_cols)
    print(f"\n{name} - Top 5 questions with highest error rates:")
    print(error_df.head(5).to_string(index=False))

# --- Compute and Output Question Difficulty Ranking ---
# Calculate difficulty as the proportion of 0's (i.e., 1 - proportion of 1's) for each question in the training data.
difficulty = 1 - df_train[d_cols].mean()
difficulty = difficulty.sort_values(ascending=False)
print("\nQuestions ranked by difficulty (most difficult at the top):")
print(difficulty.to_string())

# --- Generate Predictions with Best Performing Model ---
if best_pipeline_final is not None:
    print(f"\nBest model: {best_name} with per-label accuracy: {best_accuracy:.2f}")
    
    # Split df_predict into bound and unbound groups based on the UNBOUND column:
    # If UNBOUND == 'U', the student is unbound; otherwise, they are bound.
    df_predict_bound = df_predict[df_predict["UNBOUND"] != "U"]
    df_predict_unbound = df_predict[df_predict["UNBOUND"] == "U"]
    
    # Create prediction features for each group
    predict_features_bound = df_predict_bound[cat_cols + num_cols + e_cols]
    predict_features_unbound = df_predict_unbound[cat_cols + num_cols + e_cols]
    
    # Predict using the best pipeline
    y_pred_bound = best_pipeline_final.predict(predict_features_bound)
    y_pred_unbound = best_pipeline_final.predict(predict_features_unbound)
    
    # Build DataFrames for predictions
    pred_bound_df = pd.DataFrame(y_pred_bound, columns=d_cols)
    pred_bound_df.insert(0, "STUDENT ID", df_predict_bound["STUDENT ID"].values)
    
    pred_unbound_df = pd.DataFrame(y_pred_unbound, columns=d_cols)
    pred_unbound_df.insert(0, "STUDENT ID", df_predict_unbound["STUDENT ID"].values)
    
    # Save predictions CSV files into the data folder
    output_path_bound = os.path.join(os.path.dirname(__file__), "..", "data", f"predictions_bound_{best_name}.csv")
    output_path_unbound = os.path.join(os.path.dirname(__file__), "..", "data", f"predictions_unbound_{best_name}.csv")
    
    pred_bound_df.to_csv(output_path_bound, index=False)
    pred_unbound_df.to_csv(output_path_unbound, index=False)
    
    print(f"Predictions for bound students saved for {best_name} in '{output_path_bound}'")
    print(f"Predictions for unbound students saved for {best_name} in '{output_path_unbound}'")
    
    # --- Combine Bound and Unbound Predictions into Overall File ---
    pred_overall_df = pd.concat([pred_bound_df, pred_unbound_df], ignore_index=True)
    output_path_overall = os.path.join(os.path.dirname(__file__), "..", "data", f"predictions_overall_{best_name}.csv")
    pred_overall_df.to_csv(output_path_overall, index=False)
    print(f"Combined overall predictions saved for {best_name} in '{output_path_overall}'")

print("\nstudent_model3.py completed.")