import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import ttest_ind

print("Running data_analysis.py")

# Load the data
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "updated_student_data.csv")
df = pd.read_csv(data_path)

# Clean column names
df.columns = df.columns.str.strip().str.replace(":", "", regex=False)

# Extract class columns
class_start, class_end = "100", "729R"
class_cols = df.loc[:, class_start:class_end].columns

# Map letter grades to GPA-style values
grade_map = {
    'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7,
    'C+': 2.3, 'C': 2.0, 'C-': 1.7, 'D+': 1.3, 'D': 1.0,
    'D-': 0.7, 'F': 0.0, 'S': 2.5, 'U': 1.0, 'W': 0.0
}

# Convert letter grades to numeric scores
for col in class_cols:
    df[col] = df[col].astype(str).str.strip().map(grade_map)

# Compute Class GPA per student
df["Class GPA"] = df[class_cols].mean(axis=1, skipna=True)

# Split into training and prediction sets
df_train = df[df["DUCK TAKEN"] == 1].copy()
print(f"Initial number of rows in training set: {len(df_train)}")
df_predict = df[df["DUCK TAKEN"] == 0].copy()

# Define duck question columns (assumes columns starting with "D-")
d_cols = [col for col in df_train.columns if col.startswith("D-")]

# Function: Compare duck scores per question between unbound and non-unbound groups
def plot_duck_scores_comparison(pred_unbound, pred_bound, d_cols):
    # Calculate average score for each duck question in both groups
    unbound_avg = pred_unbound[d_cols].mean()
    bound_avg = pred_bound[d_cols].mean()
    
    # Setup a grouped bar chart
    x = range(len(d_cols))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width/2 for i in x], unbound_avg, width, label='Unbound')
    ax.bar([i + width/2 for i in x], bound_avg, width, label='Non-Unbound')
    ax.set_xticks(x)
    ax.set_xticklabels(d_cols, rotation=45, ha='right')
    ax.set_xlabel('Duck Questions')
    ax.set_ylabel('Average Score')
    ax.set_title('Duck Scores per Question: Unbound vs Non-Unbound')
    ax.legend()
    plt.tight_layout()
    plt.show()

# Load separate unbound and bound predictions and compare
pred_unbound_path = os.path.join(os.path.dirname(__file__), "..", "data", "predictions_unbound_RandomForest.csv")
pred_bound_path = os.path.join(os.path.dirname(__file__), "..", "data", "predictions_bound_RandomForest.csv")
if os.path.exists(pred_unbound_path) and os.path.exists(pred_bound_path):
    pred_unbound = pd.read_csv(pred_unbound_path)
    print(f"Loaded {len(pred_unbound)} rows in unbound predictions")
    pred_bound = pd.read_csv(pred_bound_path)
    print(f"Loaded {len(pred_bound)} rows in bound predictions")

    # Impute missing scores with column means, drop entirely empty columns
    unbound_means = pred_unbound[d_cols].mean()
    pred_unbound[d_cols] = pred_unbound[d_cols].fillna(unbound_means)
    valid_d_unbound = [col for col in d_cols if not np.isnan(unbound_means[col])]

    bound_means = pred_bound[d_cols].mean()
    pred_bound[d_cols] = pred_bound[d_cols].fillna(bound_means)
    valid_d_bound = [col for col in d_cols if not np.isnan(bound_means[col])]

    # Find questions present in both sets
    common_d_cols = [col for col in valid_d_unbound if col in valid_d_bound]
    if common_d_cols:
        plot_duck_scores_comparison(pred_unbound, pred_bound, common_d_cols)
        overall_unbound = pred_unbound[common_d_cols].mean().mean()
        overall_bound = pred_bound[common_d_cols].mean().mean()
        print(f"Overall average duck score (Unbound): {overall_unbound:.2f}")
        print(f"Overall average duck score (Non-Unbound): {overall_bound:.2f}")
    else:
        print("No common duck question columns available after dropping empty ones.")
else:
    print("Unbound or bound predictions files not found. Skipping duck scores comparison visualization.")

# Use students with actual duck scores to compute correlation
numeric_df = df_train.select_dtypes(include=[np.number])
duck_corr = numeric_df.corr().loc[:, d_cols].dropna(how='all')

# Drop rows with no variance or missing completely
duck_corr = duck_corr.dropna(how='all').replace([np.inf, -np.inf], np.nan).dropna(how='any')

# Plot a heatmap of these correlations
if not duck_corr.empty:
    plt.figure(figsize=(18, 10))
    sns.heatmap(duck_corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Between Features and Duck Test Questions (D-1 to D-60)")
    plt.tight_layout()
    plt.show()
else:
    print("No valid correlations found between features and duck questions.")

# -------------------------------
# New Section: Regression Analysis to Determine Impact on Overall Duck Scores
# -------------------------------

# Compute overall duck score for each student as the mean of duck question scores
df["OverallDuckScore"] = df[d_cols].mean(axis=1)

# Filter predictors to only include columns with sufficient data coverage
all_numeric = df.select_dtypes(include=['number'])
predictor_cols = [col for col in all_numeric.columns if col not in d_cols and col != "OverallDuckScore" and col != "DUCK TAKEN" and all_numeric[col].notna().mean() >= 0.7]

# Prepare full training set by imputing missing values instead of dropping rows
df_valid = df_train.copy()
# Impute missing predictor values with column means
df_valid[predictor_cols] = df_valid[predictor_cols].fillna(df_valid[predictor_cols].mean())
# Compute target as mean duck question score (skipping NaNs)
y = df_valid[d_cols].mean(axis=1, skipna=True)

# Ensure there is at least one predictor column
if len(predictor_cols) > 0:
    # Use only students who have not taken the duck
    min_predictors_required = int(0.6 * len(predictor_cols))
    min_duck_scores_required = int(0.6 * len(d_cols))

    df_valid = df_train.copy()
    df_valid["predictor_na"] = df_valid[predictor_cols].isna().sum(axis=1)
    df_valid["duck_na"] = df_valid[d_cols].isna().sum(axis=1)

    df_valid = df_valid[
        (df_valid["predictor_na"] <= len(predictor_cols) - min_predictors_required) &
        (df_valid["duck_na"] <= len(d_cols) - min_duck_scores_required)
    ].copy()

    df_valid.drop(columns=["predictor_na", "duck_na"], inplace=True)

    # Check for any remaining NaNs in predictor columns
    nan_summary = df_valid[predictor_cols].isna().sum()
    print("Remaining NaNs in predictors after filtering:")
    print(nan_summary[nan_summary > 0])

    # Fill remaining NaNs in predictor columns with column means
    df_valid[predictor_cols] = df_valid[predictor_cols].fillna(df_valid[predictor_cols].mean())

    if df_valid.empty:
        print("No students with complete predictor data for regression analysis.")
    else:
        # Standardize predictors to compare coefficients
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_valid[predictor_cols])

        # Fit a linear regression model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        print("Duck score training mean:", y.mean())
        print("Model coefficients:")
        print(pd.Series(model.coef_, index=predictor_cols).sort_values())

        # Create a DataFrame of coefficients
        coef_df = pd.DataFrame({
            'Feature': predictor_cols,
            'Coefficient': model.coef_
        })

        # Sort by absolute coefficient value (impact magnitude)
        coef_df['AbsCoefficient'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values(by='AbsCoefficient', ascending=True)

        # Plot a horizontal bar chart of standardized coefficients
        plt.figure(figsize=(10, 6))
        plt.barh(coef_df['Feature'], coef_df['Coefficient'])
        plt.xlabel("Standardized Coefficient")
        plt.title("Impact of Predictors on Overall Duck Score")
        plt.tight_layout()
        plt.show()

        #-----------------------------------------------------------------------------  
        # Predict duck score for all students (imputing missing predictors and scaling)
        df_all = df.copy()
        # Impute missing predictor values using the means computed on df_valid
        train_means = df_valid[predictor_cols].mean()
        df_all[predictor_cols] = df_all[predictor_cols].fillna(train_means)

        # Scale predictors using the same scaler
        X_all_scaled = scaler.transform(df_all[predictor_cols])
        df_all["PredictedDuckScore"] = model.predict(X_all_scaled)

        # Compare predicted distributions for all students by UNBOUND status
        unbound_pred_all = df_all[df_all["UNBOUND"] == "U"]["PredictedDuckScore"]
        bound_pred_all = df_all[df_all["UNBOUND"] != "U"]["PredictedDuckScore"]

        print(f"Total students with predictions: {len(df_all)}")
        print(f"Unbound predictions count: {len(unbound_pred_all)}")
        print(f"Non-Unbound predictions count: {len(bound_pred_all)}")

        # Plot histogram of predicted duck scores for all students
        plt.figure(figsize=(10, 6))
        plt.hist(unbound_pred_all, bins=20, alpha=0.6, label="Unbound", density=True)
        plt.hist(bound_pred_all, bins=20, alpha=0.6, label="Non-Unbound", density=True)
        plt.title("Distribution of Predicted Duck Scores by UNBOUND Status (All Students)")
        plt.xlabel("Predicted Duck Score")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Print mean predicted scores for all students
        print(f"Mean predicted duck score (Unbound): {unbound_pred_all.mean():.2f}")
        print(f"Mean predicted duck score (Non-Unbound): {bound_pred_all.mean():.2f}")

        # Perform independent t-test between unbound and bound predicted scores
        t_stat, p_val = ttest_ind(unbound_pred_all, bound_pred_all, equal_var=False)
        print(f"T-test statistic: {t_stat:.4f}")
        print(f"P-value: {p_val:.4e}")
        if p_val < 0.05:
            print("The difference in predicted duck scores between Unbound and Non-Unbound is statistically significant (p < 0.05).")
        else:
            print("The difference in predicted duck scores is not statistically significant (p >= 0.05).")

        from scipy.stats import sem, t

        # Compute confidence interval for difference in predicted means
        diff_mean = unbound_pred_all.mean() - bound_pred_all.mean()
        combined_se = np.sqrt(unbound_pred_all.var(ddof=1)/len(unbound_pred_all) + bound_pred_all.var(ddof=1)/len(bound_pred_all))
        df_combined = len(unbound_pred_all) + len(bound_pred_all) - 2
        ci_low, ci_high = t.interval(0.95, df_combined, loc=diff_mean, scale=combined_se)
        print(f"95% Confidence interval for mean difference: ({ci_low:.4f}, {ci_high:.4f})")

        # -----------------------------------------------------------------------------  
        # Statistical comparison of predicted DUCK scores between groups
        def cohens_d(x, y):
            nx, ny = len(x), len(y)
            pooled_std = np.sqrt(((nx - 1)*x.std(ddof=1) + (ny - 1)*y.std(ddof=1)) / (nx + ny - 2))
            return (x.mean() - y.mean()) / pooled_std

        d_effect_pred = cohens_d(unbound_pred_all, bound_pred_all)

        print(f"\nEffect size (Cohen's d) for predicted scores: {d_effect_pred:.4f}")
else:
    print("No suitable predictor features found for regression analysis.")