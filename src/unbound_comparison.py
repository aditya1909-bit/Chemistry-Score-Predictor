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
from sklearn.metrics import make_scorer, hamming_loss

# --- Custom Scorer ---
def per_label_accuracy_score(y_true, y_pred):
    return (y_true == y_pred).mean().mean()

scorer = make_scorer(per_label_accuracy_score, greater_is_better=True)

# --- Data Loading ---
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student_data.csv")
df = pd.read_csv(data_path)

# Split the data:
# - df_train: Students who took the D test (DUCK TAKEN == 1)
# - df_predict: Students who haven't taken the D test (DUCK TAKEN == 0)
df_train = df[df["DUCK TAKEN"] == 1].copy()
df_predict = df[df["DUCK TAKEN"] == 0].copy()

# Clean column names: remove extra spaces and colons
df_train.columns = df_train.columns.str.strip().str.replace(":", "", regex=False)
df_predict.columns = df_predict.columns.str.strip().str.replace(":", "", regex=False)

# For training data, drop identifier columns not used as features.
df_train.drop(columns=["STUDENT ID", "DUCK TAKEN"], inplace=True)

# --- Define Column Groups ---
# D columns (D-1 to D-60) are our binary targets.
d_cols = [col for col in df_train.columns if col.startswith("D-")]

# E test columns (E0, E1, E2) will be used as features.
e_cols = [col for col in df_train.columns if col.startswith("E0-") or 
          col.startswith("E1-") or col.startswith("E2-")]

# Other numerical columns
num_cols = ["HS GPA", "SAT MATH", "SAT ENG", "SAT", "EMORY GPA", "CHEM GPA"]

# Categorical columns (including UNBOUND for later grouping)
cat_cols = ["UNBOUND", "ETHNICITY", "GENDER"]

# --- Prepare Features and Targets ---
X = df_train[cat_cols + num_cols + e_cols]
y = df_train[d_cols]

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

# --- Model Training using RandomForest with GridSearchCV ---
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", MultiOutputClassifier(RandomForestClassifier(random_state=42)))
])

param_grid = {
    'classifier__estimator__n_estimators': [200, 300],
    'classifier__estimator__max_depth': [None, 20, 30]
}

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=3,
    scoring=scorer,
    verbose=0
)
grid.fit(X_train, y_train)
best_pipeline = grid.best_estimator_
print("Best parameters:", grid.best_params_)

# Evaluate on validation set
y_pred_val = best_pipeline.predict(X_val)
val_per_label_accuracy = per_label_accuracy_score(y_val, y_pred_val)
print("Validation average per-label accuracy: {:.2f}".format(val_per_label_accuracy))

# --- Predictions for Unbound Comparison ---
# Use best_pipeline to predict D test outcomes for students who haven't taken the test.
# We need the UNBOUND column for grouping, so we keep it.
predict_features = df_predict[cat_cols + num_cols + e_cols]
y_pred = best_pipeline.predict(predict_features)

# Build a predictions DataFrame with STUDENT ID and UNBOUND column
pred_df = pd.DataFrame(y_pred, columns=d_cols)
pred_df.insert(0, "STUDENT ID", df_predict["STUDENT ID"].values)
pred_df["UNBOUND"] = df_predict["UNBOUND"].values

# Save overall predictions file (will be ignored by git if data/ is in .gitignore)
output_path_overall = os.path.join(os.path.dirname(__file__), "..", "data", "predictions_overall.csv")
pred_df.to_csv(output_path_overall, index=False)
print("Overall predictions saved to:", output_path_overall)

# --- Split Predictions by UNBOUND Group ---
pred_unbound = pred_df[pred_df["UNBOUND"] == "U"].copy()
pred_bound = pred_df[pred_df["UNBOUND"] != "U"].copy()

# Compute overall average predicted score for each group (average across all D questions)
overall_avg_unbound = pred_unbound[d_cols].mean().mean()
overall_avg_bound = pred_bound[d_cols].mean().mean()

print("\nOverall average predicted score:")
print("Unbound group: {:.2f}".format(overall_avg_unbound))
print("Non-unbound group: {:.2f}".format(overall_avg_bound))

# For each group, rank questions by difficulty.
# Difficulty is defined as: 1 - (average predicted score for the question).
# Lower predicted scores indicate that the question is generally answered incorrectly.
difficulty_unbound = 1 - pred_unbound[d_cols].mean()
difficulty_bound = 1 - pred_bound[d_cols].mean()

difficulty_unbound = difficulty_unbound.sort_values(ascending=False)  # most difficult at top
difficulty_bound = difficulty_bound.sort_values(ascending=False)

print("\nQuestions that unbound students would generally get wrong (ranked by difficulty):")
print(difficulty_unbound.to_string())

print("\nQuestions that non-unbound students would generally get wrong (ranked by difficulty):")
print(difficulty_bound.to_string())

# Save group-specific predictions
output_path_unbound = os.path.join(os.path.dirname(__file__), "..", "data", "predictions_unbound.csv")
output_path_bound = os.path.join(os.path.dirname(__file__), "..", "data", "predictions_bound.csv")
pred_unbound.to_csv(output_path_unbound, index=False)
pred_bound.to_csv(output_path_bound, index=False)
print("\nPredictions for unbound saved to:", output_path_unbound)
print("Predictions for non-unbound saved to:", output_path_bound)
