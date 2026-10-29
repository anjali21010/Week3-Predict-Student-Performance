# Week 3 – Predict Student Performance

## Project objective
This beginner-friendly Machine Learning project uses **Linear Regression** to predict a student's Performance Index from study-related features.

## Dataset
Download the Student Performance dataset CSV from Kaggle. Save the downloaded CSV in the same folder as `week3_student_performance.py` and rename it:

`Student_Performance.csv`

The CSV must contain these five columns (capitalization and spaces/underscores are handled by the script):

- `Hours_studied`
- `Previous_scores`
- `Sleep_hours`
- `sample_question_papers_practiced`
- `performance_index`

## Requirements
- Python 3
- pandas
- matplotlib
- scikit-learn

## How to run (Windows / VS Code)
1. Extract this project ZIP.
2. Download the CSV dataset from Kaggle and put it in the extracted project folder.
3. Open that folder in VS Code.
4. Open **Terminal → New Terminal**.
5. Install the packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
6. Run the project:
   ```bash
   python week3_student_performance.py
   ```

## Tasks completed
- Loads and checks the dataset
- Cleans missing or non-numeric values in the required columns
- Splits data into 80% training and 20% testing
- Trains a Scikit-learn Linear Regression model
- Predicts the Performance Index for test data
- Evaluates the model using R² Score and Mean Absolute Error (MAE)
- Saves prediction results and graphs

## Output files
The script creates an `outputs` folder containing:
- `predicted_results.csv` – actual and predicted values with absolute error
- `actual_vs_predicted.png` – graph comparing actual and predicted values
- `feature_coefficients.png` – model coefficient graph
- `model_metrics.txt` – dataset sizes and evaluation metrics

## Understanding the metrics
- **R² Score:** Indicates how well the model explains variation in the target values. It is not the same as classification accuracy.
- **MAE:** Average absolute difference between actual and predicted Performance Index values. Lower is generally better.

## Important
The project ZIP does not include the Kaggle dataset. Download it yourself and keep it in the project folder. Do not invent or manually report model scores; run the script on the real dataset and use the generated metrics.

## GitHub upload
Create a new public repository named `Week3-Predict-Student-Performance`. Upload the project files and the dataset only if the dataset's license permits redistribution. Do not upload the `outputs` folder before running the project unless you want to include your real generated results.
