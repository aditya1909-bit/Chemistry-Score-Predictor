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


print("Running unbound_comparison.py")

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

# --- Load Predictions from CSV Files for Unbound Comparison ---
output_path_bound_rf = os.path.join(os.path.dirname(__file__), "..", "data", "predictions_bound_RandomForest.csv")
output_path_unbound_rf = os.path.join(os.path.dirname(__file__), "..", "data", "predictions_unbound_RandomForest.csv")
pred_bound = pd.read_csv(output_path_bound_rf)
pred_unbound = pd.read_csv(output_path_unbound_rf)
pred_df = pd.concat([pred_bound, pred_unbound], ignore_index=True)
print("Loaded predictions from predictions_bound_RandomForest.csv and predictions_unbound_RandomForest.csv")

# Compute overall average predicted score for each group (average across all D questions)
overall_avg_unbound = pred_unbound[d_cols].mean().mean()
overall_avg_bound = pred_bound[d_cols].mean().mean()

print("\nOverall average predicted score:")
print("Unbound group: {:.2f}".format(pred_unbound[d_cols].mean().mean()))
print("Non-unbound group: {:.2f}".format(pred_bound[d_cols].mean().mean()))

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

# Save group-specific predictions (if needed)
output_path_unbound_new = os.path.join(os.path.dirname(__file__), "..", "data", "predictions_unbound_analysis.csv")
output_path_bound_new = os.path.join(os.path.dirname(__file__), "..", "data", "predictions_bound_analysis.csv")
pred_unbound.to_csv(output_path_unbound_new, index=False)
pred_bound.to_csv(output_path_bound_new, index=False)
print("\nPredictions for unbound saved to:", output_path_unbound_new)
print("Predictions for non-unbound saved to:", output_path_bound_new)

print("unbound_comparison.py completed successfully.")
