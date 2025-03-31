# Chemistry Score Predictor

This project uses machine learning to predict students' final chemistry exam scores based on their performance in earlier tests and other relevant features.

## 📊 Project Overview

The goal is to develop a predictive model that estimates a student's final score on the DUCK exam (`D`) in chemistry using earlier exam scores (`E0`, `E1`, `E2`) and demographic or performance-based features. This can be useful for identifying at-risk students and providing early interventions.

## 📁 Project Structure

```
Chemistry-Score-Predictor/
├── data/                  # Raw and cleaned datasets
├── notebooks/             # Jupyter notebooks for EDA and model development
├── models/                # Trained models and model evaluation reports
├── src/                   # Source code for preprocessing and training
├── README.md              # Project documentation
```

## 🔍 Features Used

- `E0`, `E1`, `E2`: Scores from earlier chemistry tests. Here `E0` is the ECCI Test that Emory students took before matriculation, `E1` is the ECCI Test that they took after Chem 204 in the new curriculum, and Organic Chem 2 in the old curriculum, and `E2` is the ECCI that senior chem majors take.
- Demographic information (e.g., age, gender, etc.)
- Other performance indicators (e.g., attendance, participation).

## 🧠 Models

The project explores different regression models including:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting
- Neural Networks

Model performance is evaluated using metrics such as MAE, RMSE, and R² score.

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Chemistry-Score-Predictor.git
   cd Chemistry-Score-Predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the model training script:
   ```bash
   python src/train_model.py
   ```

## 🧪 Testing & Evaluation

Evaluation results, plots, and comparison tables are available in the `notebooks/` and `models/` directories. You can also generate predictions for new student data using:

```bash
python src/predict.py --input new_data.csv
```

## 📈 Example Results

| Model              | MAE   | RMSE  | R²     |
|-------------------|-------|-------|--------|
| Linear Regression | 5.2   | 6.3   | 0.76   |
| Random Forest     | 4.1   | 5.0   | 0.85   |

## 📌 TODO

- Add support for cross-validation.
- Integrate a simple web UI for predictions.
- Expand feature set with behavioral metrics.

## 👩‍🔬 Author

Aditya Dutta

## 📄 License

This project is licensed under the MIT License.
