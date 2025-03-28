import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("student_data.csv")

# Split into two sets: those who took the D test (DUCK TAKEN == 1) and those who didn't (DUCK TAKEN == 0)
df_train = df[df["DUCK TAKEN"] == 1].copy()   # for training and model evaluation
df_predict = df[df["DUCK TAKEN"] == 0].copy()   # for later prediction

# For training data, drop the identifier and DUCK TAKEN columns
df_train.drop(columns=["STUDENT ID", "DUCK TAKEN"], inplace=True)

# Identify columns:
# D test columns are our multi-output targets.
d_cols = [col for col in df_train.columns if col.startswith("D-")]
# E test columns (there are three groups: E0, E1, E2)
e_cols = [col for col in df_train.columns if col.startswith("E0-") or 
          col.startswith("E1-") or col.startswith("E2-")]
# Other numerical columns
num_cols = ["HS GPA", "SAT MATH", "SAT ENG", "SAT", "EMORY GPA", "CHEM GPA"]
# Categorical columns
cat_cols = ["UNBOUND:", "ETHNICITY", "GENDER"]

# Define features and labels for training
X = df_train[cat_cols + num_cols + e_cols]
y = df_train[d_cols]  # Multi-output targets

# Preprocessing pipeline:
# - Numeric: Impute missing values (using mean) and scale.
# - Categorical: Impute missing (most frequent) and one-hot encode.
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

# Define three different models using MultiOutputRegressor
models = {
    "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
    "XGBoost": MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
    "NeuralNetwork": MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42))
}

# Split training data into train and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Loop through each model, train, evaluate, and then predict for students who didn't take the D test.
for name, model in models.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Validate the model
    y_pred_val = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_val))
    print(f"{name} RMSE on validation set: {rmse:.2f}")
    
    # Prepare the prediction dataset:
    # For df_predict, we need the same features.
    # Keep STUDENT ID for reference and drop DUCK TAKEN if present.
    predict_features = df_predict[cat_cols + num_cols + e_cols]
    
    # Predict D test scores for students who haven't taken the test
    y_pred = pipeline.predict(predict_features)
    
    # Create a DataFrame with predictions; retain the STUDENT ID for identification.
    pred_df = pd.DataFrame(y_pred, columns=d_cols)
    pred_df.insert(0, "STUDENT ID", df_predict["STUDENT ID"].values)
    
    # Save predictions to a CSV file
    output_filename = f"predictions_{name}.csv"
    pred_df.to_csv(output_filename, index=False)
    print(f"Predictions saved for {name} in '{output_filename}'")