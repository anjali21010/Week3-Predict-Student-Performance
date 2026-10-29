from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Keep the CSV in the same folder as this Python file.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "Student_Performance.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Dataset headers can have spaces or underscores. Normalize them first.
COLUMN_MAP = {
    "hours studied": "Hours_studied",
    "hours_studied": "Hours_studied",
    "previous scores": "Previous_scores",
    "previous_scores": "Previous_scores",
    "sleep hours": "Sleep_hours",
    "sleep_hours": "Sleep_hours",
    "sample question papers practiced": "sample_question_papers_practiced",
    "sample_question_papers_practiced": "sample_question_papers_practiced",
    "performance index": "performance_index",
    "performance_index": "performance_index",
}

FEATURES = [
    "Hours_studied",
    "Previous_scores",
    "Sleep_hours",
    "sample_question_papers_practiced",
]
TARGET = "performance_index"


def main():
    if not DATA_FILE.exists():
        print("\nDATASET NOT FOUND")
        print(f"Please download the Student Performance CSV from Kaggle and save it here:\n{DATA_FILE}")
        print("Rename the CSV file to: Student_Performance.csv")
        input("\nPress Enter to close...")
        return

    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as exc:
        print(f"Could not read the CSV file: {exc}")
        input("\nPress Enter to close...")
        return

    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns=COLUMN_MAP)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        print("\nThe CSV does not contain the required columns.")
        print("Missing columns:", ", ".join(missing))
        print("Columns found:", ", ".join(df.columns))
        print("Please use the Student Performance dataset with the five columns listed in README.md.")
        input("\nPress Enter to close...")
        return

    # Convert required fields to numbers and remove incomplete rows.
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=FEATURES + [TARGET]).copy()
    removed = before - len(df)

    if len(df) < 5:
        print("Not enough valid rows to train the model. Please check your CSV dataset.")
        input("\nPress Enter to close...")
        return

    X = df[FEATURES]
    y = df[TARGET]

    # Keep the test set separate so evaluation uses unseen rows.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    results = X_test.copy()
    results["Actual_performance_index"] = y_test.values
    results["Predicted_performance_index"] = predictions.round(2)
    results["Absolute_error"] = (
        results["Actual_performance_index"] - results["Predicted_performance_index"]
    ).abs().round(2)
    results.to_csv(OUTPUT_DIR / "predicted_results.csv", index=False)

    # Plot actual values against predicted values.
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.6)
    low = min(float(y_test.min()), float(predictions.min()))
    high = max(float(y_test.max()), float(predictions.max()))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual Performance Index")
    plt.ylabel("Predicted Performance Index")
    plt.title("Actual vs Predicted Student Performance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # Plot model coefficients to show how the model uses each feature.
    plt.figure(figsize=(9, 5))
    plt.bar(FEATURES, model.coef_)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Linear Regression Coefficient")
    plt.title("Feature Coefficients")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_coefficients.png", dpi=150)
    plt.close()

    metrics = (
        "WEEK 3 - PREDICT STUDENT PERFORMANCE\n"
        "===================================\n"
        f"Valid dataset rows: {len(df)}\n"
        f"Rows removed due to missing/non-numeric values: {removed}\n"
        f"Training rows: {len(X_train)}\n"
        f"Testing rows: {len(X_test)}\n"
        f"R2 Score: {r2:.4f}\n"
        f"Mean Absolute Error (MAE): {mae:.4f}\n\n"
        "Model: Linear Regression\n"
        "Train-test split: 80% training / 20% testing\n"
        "Random state: 42\n"
        "\nNote: R2 is a goodness-of-fit score, not classification accuracy.\n"
        "MAE is the average absolute prediction error in Performance Index units.\n"
    )
    (OUTPUT_DIR / "model_metrics.txt").write_text(metrics, encoding="utf-8")

    print("\nPROJECT COMPLETED SUCCESSFULLY!")
    print("--------------------------------")
    print(f"Valid dataset rows: {len(df)}")
    print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"\nResults saved in: {OUTPUT_DIR}")
    print("- predicted_results.csv")
    print("- actual_vs_predicted.png")
    print("- feature_coefficients.png")
    print("- model_metrics.txt")
    input("\nPress Enter to close...")


if __name__ == "__main__":
    main()
