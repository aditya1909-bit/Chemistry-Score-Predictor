import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

print("Running data_analysis.py")

# Load the data
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "student_data.csv")
df = pd.read_csv(data_path)

# Clean column names
df.columns = df.columns.str.strip().str.replace(":", "", regex=False)

# Split into training and prediction sets
df_train = df[df["DUCK TAKEN"] == 1].copy()
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

# Load predicted results and display the duck scores comparison graph
pred_path = os.path.join(os.path.dirname(__file__), "..", "data", "predictions_overall.csv")
if os.path.exists(pred_path):
    pred_df = pd.read_csv(pred_path)
    
    # Split the predictions by UNBOUND status
    pred_unbound = pred_df[pred_df["UNBOUND"] == "U"]
    pred_bound = pred_df[pred_df["UNBOUND"] != "U"]
    
    # Plot the grouped bar chart comparing duck scores for each question
    plot_duck_scores_comparison(pred_unbound, pred_bound, d_cols)
    
    # Optionally, print overall average scores for both groups
    overall_unbound = pred_unbound[d_cols].mean().mean()
    overall_bound = pred_bound[d_cols].mean().mean()
    print(f"Overall average duck score (Unbound): {overall_unbound:.2f}")
    print(f"Overall average duck score (Non-Unbound): {overall_bound:.2f}")
else:
    print("Predictions file not found. Skipping duck scores comparison visualization.")

# Plot a clustered heatmap for all numeric factors in the dataset
numeric_cols = df.select_dtypes(include=['number']).columns
corr_matrix = df[numeric_cols].corr()

clustergrid = sns.clustermap(
    corr_matrix, 
    method='average',      
    metric='euclidean',    
    cmap='coolwarm', 
    figsize=(15, 12),      
    annot=False            
)
clustergrid.ax_heatmap.set_title("Correlation Clustermap of All Numeric Factors", pad=20)
plt.show()

# -------------------------------
# New Section: Regression Analysis to Determine Impact on Overall Duck Scores
# -------------------------------

# Compute overall duck score for each student as the mean of duck question scores
df["OverallDuckScore"] = df[d_cols].mean(axis=1)

# Identify potential predictor features: all numeric columns except duck scores and the overall score
predictor_cols = [col for col in numeric_cols if col not in d_cols and col != "OverallDuckScore" and col != "DUCK TAKEN"]

# Ensure there is at least one predictor column
if len(predictor_cols) > 0:
    X = df[predictor_cols].dropna()  # drop rows with missing predictor values
    y = df.loc[X.index, "OverallDuckScore"]

    # Standardize predictors to compare coefficients
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_scaled, y)

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

    print("Regression analysis complete. The chart displays the standardized impact of each predictor on the overall duck score.")
else:
    print("No suitable predictor features found for regression analysis.")