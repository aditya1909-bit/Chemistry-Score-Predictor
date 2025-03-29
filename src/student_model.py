import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import hamming_loss
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
# Build the file path to your data file in the data folder
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student_data.csv")
df = pd.read_csv(data_path)

# Split into two sets:
# - df_train: Students who took the D test (DUCK TAKEN == 1)
# - df_predict: Students who have not taken the D test (DUCK TAKEN == 0)
df_train = df[df["DUCK TAKEN"] == 1].copy()
df_predict = df[df["DUCK TAKEN"] == 0].copy()

# Clean up column names: remove extra spaces and colons
df_train.columns = df_train.columns.str.strip().str.replace(":", "", regex=False)
df_predict.columns = df_predict.columns.str.strip().str.replace(":", "", regex=False)

# For training data, drop identifier columns that won't be used as features.
df_train.drop(columns=["STUDENT ID", "DUCK TAKEN"], inplace=True)

# Define column groups
# D columns: our binary targets (values between 0 and 1)
d_cols = [col for col in df_train.columns if col.startswith("D-")]

# E test columns (E0, E1, E2) will be used as part of the features.
e_cols = [col for col in df_train.columns if col.startswith("E0-") or 
          col.startswith("E1-") or col.startswith("E2-")]

# Other numerical columns
num_cols = ["HS GPA", "SAT MATH", "SAT ENG", "SAT", "EMORY GPA", "CHEM GPA"]

# Categorical columns
cat_cols = ["UNBOUND", "ETHNICITY", "GENDER"]

# Prepare features (X) and multi-output binary labels (y)
X = df_train[cat_cols + num_cols + e_cols]
y = df_train[d_cols]

# Preprocessing pipeline:
#  - Numeric: Impute missing values (mean) and scale.
#  - Categorical: Impute missing (most frequent) and one-hot encode.
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

# Define multi-output classifiers:
models = {
    "RandomForest": MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
    "XGBoost": MultiOutputClassifier(
        XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42,
                      use_label_encoder=False, eval_metric='logloss')
    ),
    "NeuralNetwork": MultiOutputClassifier(
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    )
}

# Split training data for evaluation (e.g., 80% train, 20% validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Loop through each model, train, evaluate, and then predict on df_predict
for name, model in models.items():
    print(f"\nTraining {name}...")
    # Create a pipeline that includes preprocessing and the classifier
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model on the validation set:
    # Subset accuracy (all labels must be correct)
    subset_accuracy = pipeline.score(X_test, y_test)
    
    # Predict on validation set for alternative metrics:
    y_pred_val = pipeline.predict(X_test)
    # Hamming loss: fraction of labels mispredicted
    ham_loss = hamming_loss(y_test, y_pred_val)
    # Per-label accuracy (average across all labels)
    per_label_accuracy = (y_test == y_pred_val).mean().mean()
    
    print(f"{name} subset accuracy on validation set: {subset_accuracy:.2f}")
    print(f"{name} Hamming loss on validation set: {ham_loss:.2f}")
    print(f"{name} average per-label accuracy: {per_label_accuracy:.2f}")
    
    # Prepare features for prediction on students who haven't taken the D test.
    predict_features = df_predict[cat_cols + num_cols + e_cols]
    
    # Predict D test outcomes (binary outputs)
    y_pred = pipeline.predict(predict_features)
    
    # Build a DataFrame for the predictions and include STUDENT ID for reference.
    pred_df = pd.DataFrame(y_pred, columns=d_cols)
    pred_df.insert(0, "STUDENT ID", df_predict["STUDENT ID"].values)
    
    # Save the predictions CSV into the data folder (data/ is git-ignored)
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", f"predictions_{name}.csv")
    pred_df.to_csv(output_path, index=False)
    print(f"Predictions saved for {name} in '{output_path}'")